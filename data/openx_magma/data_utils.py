import os

import cv2
import torch
import torchvision
import yaml

from data.conversations import Constructor
from data.openx.action_tokenizer import ActionTokenizer
from data.utils.som_tom import som_prompting
from data.utils.visual_trace import visual_trace


class OpenXMagma(Constructor):
    def __init__(self, **kwargs):
        super(OpenXMagma, self).__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)
        self.task = kwargs.get('task', 'agent')
        self.use_som_tom = kwargs.get('mm_use_som_tom', True)

        tokenizer = kwargs.get('tokenizer', None)
        assert tokenizer, "Tokenizer is not provided"

        if self.mm_use_image_start_end:
            self.image_placeholder = '<image_start><image><image_end>\n'
        else:
            self.image_placeholder = '<image>\n'

        self.action_tokenizer = ActionTokenizer(tokenizer)
        self.trace_width = 256
        self.trace_height = 256

    def __call__(self, item, video_path, visual_traces, width=512, height=512):

        item['num_crops'] = 1
        
        if video_path is None and visual_trace is None:
            dummy_conversations = []
            dummy_conversations.append({'from': 'human', 'value': f"{self.image_placeholder}\nWhat is in this image?"})
            dummy_conversations.append({'from': 'gpt', 'value': "This is a blank image."})            
            item['conversations'] = dummy_conversations
            item['image_data'] = None  
            return item
        
        frame_start, frame_end = item['frame_index'], item['frame_index'] + 16  
        task_description = item['lang']
        
        gpt_response = task_description

        if self.mm_use_image_history:
            # randomly sample at most 7 unique indices in range [0, frame_start) with probability 0.3
            # if torch.rand(1).item() < 0.5:
            #     frame_idx = torch.randperm(frame_start)[:7].sort().values.tolist() + [frame_start]
            # else:
            frame_idx = [frame_start]
        else:
            frame_idx = [frame_start]
        
        item['image_data'] = self._get_frames_with_idx(video_path, frame_idx, (width, height))
        # conversation 1: Q: to do the task, what should be the next action? A: next action
        image_placeholder = ''.join([self.image_placeholder]*len(item['image_data']))
        item['conversations'] = [ 
            {"from": "human", "value": f"{image_placeholder}\nWhat action should the robot take to {gpt_response}?"},
            {"from": "gpt", "value": ''.join(['<action>']*7)}, # placeholder for action tokens
        ]
        action = visual_traces['action']
        action_token_ids = self.action_tokenizer.encode_actions_to_token_ids(action)
        item['action_token_ids'] = action_token_ids

        # conversation 2: Q: what is the robot doing? A: task description        
        # conv_user, conv_gpt, gpt_response_todo = self._construct_conv_semantic(item, gpt_response)    
        # conversations.append({'from': 'human', 'value': conv_user})
        # conversations.append({'from': 'gpt', 'value': conv_gpt})
        # item['image_data'].append(self._get_frame(video_path, frame_start, 0, (width, height)))

        if not self.use_som_tom:
            return item

        if visual_traces is None:
            return item
        
        if item['dataset_name'].decode('utf-8') in ["berkeley_cable_routing", "kuka"]:
            return item
        
        visual_traces['pred_tracks'], visual_traces['pred_visibility'] = visual_traces['trace_info']

        if width != self.trace_width:
            visual_traces['pred_tracks'][...,0] =  visual_traces['pred_tracks'][...,0] * width // self.trace_width
        if height != self.trace_height:
            visual_traces['pred_tracks'][...,1] =  visual_traces['pred_tracks'][...,1] * height // self.trace_height

        if len(visual_traces['pred_tracks'].shape) == 3:
            visual_traces['pred_tracks'] = visual_traces['pred_tracks'][None]
        if len(visual_traces['pred_visibility'].shape) == 2:
            visual_traces['pred_visibility'] = visual_traces['pred_visibility'][None]

        frame_pos = 0
        pred_tracks = visual_traces['pred_tracks'][:, frame_pos:]
        pred_visibility = visual_traces['pred_visibility'][:, frame_pos:]
        step_to_predict = pred_tracks.size(1)

        if step_to_predict == 0:            
            return item
        
        pred_tracks_history = visual_traces['pred_tracks'][:, :max(1, frame_pos+1)]
        pred_visibility_history = visual_traces['pred_visibility'][:, :max(1, frame_pos+1)]

        # only keep points that are visible at at least half steps
        valid_idx = pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]

        if valid_idx.sum() <= 1:
            image = self._get_frame(video_path, frame_start, 0, (width, height))            
            conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces, 0)
            item['conversations'].append({'from': 'human', 'value': conv_user})
            item['conversations'].append({'from': 'gpt', 'value': conv_gpt})
            item['image_data'].append(image)
            return item

        pred_tracks = pred_tracks[:, :, valid_idx]
        pred_visibility = pred_visibility[:, :, valid_idx]
        pred_tracks_history = pred_tracks_history[:, :, valid_idx]
        pred_visibility_history = pred_visibility_history[:, :, valid_idx]
        
        # calculate the trajectory lenght for pred_tracks
        pred_tracks_length = self.trace.visual_trace_length(pred_tracks, pred_visibility, (1, 1)).squeeze(0)
        # if 80% of the pred_tracks_length is larger than 2, then there is camera motion
        camera_motion = (pred_tracks_length > 1).sum() > 0.8*pred_tracks_length.size(0)
        camera_motion = True if item['dataset_tag'] in ['ego4d', 'epic'] else camera_motion
        
        start_pos = pred_tracks[:, 0][0]
        reference_pts_np = start_pos.cpu().numpy().reshape(-1, 2)

        if camera_motion:
            # remove camera motion using homography transformation
            try:
                future_pts_transformed = []
                for k in range(1, pred_tracks.shape[1]):
                    future_pts = pred_tracks[:, k][0]
                    future_pts_np = future_pts.cpu().numpy().reshape(-1, 2)
                    try:
                        (H, status) = cv2.findHomography(future_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
                    except Exception as e:
                        continue
                    future_pts_np_transformed = cv2.perspectiveTransform(future_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
                    future_pts_transformed_k = torch.tensor(future_pts_np_transformed, dtype=torch.float32)
                    future_pts_transformed.append(future_pts_transformed_k)            
                pred_tracks = torch.stack([start_pos] + future_pts_transformed, dim=0).unsqueeze(0)           
            except Exception as e:
                pass
            
            if pred_tracks_history.size(1) > 0:
                try:
                    history_pts_transformed = []
                    for k in range(0, pred_tracks_history.shape[1]):
                        history_pts = pred_tracks_history[:, k][0]
                        history_pts_np = history_pts.cpu().numpy().reshape(-1, 2)
                        try:
                            (H, status) = cv2.findHomography(history_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
                        except Exception as e:
                            continue
                        history_pts_np_transformed = cv2.perspectiveTransform(history_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
                        history_pts_transformed_k = torch.tensor(history_pts_np_transformed, dtype=torch.float32)
                        history_pts_transformed.append(history_pts_transformed_k)            
                    pred_tracks_history = torch.stack(history_pts_transformed, dim=0).unsqueeze(0)   
                except Exception as e:
                    pass
        
        # step 2: find positive traces and negative traces
        track_length = self.trace.visual_trace_length(pred_tracks, pred_visibility, (1, 1)).squeeze(0)
        threshold = 1 # max(track_length.max(), 2) * self.settings['trace_processor']['postive_factor_threshold']
        # video is almost static
        if (track_length > threshold).sum() <= 1:
            image = self._get_frame(video_path, frame_start, 0, (width, height))            
            conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces, 0)
            item['conversations'].append({'from': 'human', 'value': conv_user})
            item['conversations'].append({'from': 'gpt', 'value': conv_gpt})
            item['image_data'].append(image)
            return item
        else:
            # find the positive traces and negative traces
            pos_tracks = pred_tracks[:, :, track_length > threshold]
            pos_visibility = pred_visibility[:, :, track_length > threshold]
            pos_tracks_history = pred_tracks_history[:, :, track_length > threshold]
            pos_visibility_history = pred_visibility_history[:, :, track_length > threshold]

            neg_tracks = pred_tracks[:, :, track_length <= threshold]
            neg_tracks_history = pred_tracks_history[:, :, track_length <= threshold]

            # clustering for positive traces
            
            # randome sample a number between 2 and self.num_clusters
            num_clusters_pos = torch.randint(2, 5, (1,)).item()
            pos_sampled_ids = self.trace.cluster_traces_kmeans(pos_tracks, n_clusters=num_clusters_pos, positive=True)
            if pos_sampled_ids is None:
                image = self._get_frame(video_path, frame_start, 0, (width, height))            
                conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces, 0)
                item['conversations'].append({'from': 'human', 'value': conv_user})
                item['conversations'].append({'from': 'gpt', 'value': conv_gpt})
                item['image_data'].append(image)
                return item

            pos_tracks = pos_tracks[:, :, pos_sampled_ids.bool()]
            pos_visibility = pos_visibility[:, :, pos_sampled_ids.bool()]
            pos_tracks_history = pos_tracks_history[:, :, pos_sampled_ids.bool()]
            pos_visibility_history = pos_visibility_history[:, :, pos_sampled_ids.bool()]

            # clustering for negative traces
            num_clusters_neg = torch.randint(4, 10, (1,)).item()
            neg_sampled_ids = self.trace.cluster_traces_kmeans(neg_tracks, n_clusters=num_clusters_neg)
            if neg_sampled_ids is None:
                image = self._get_frame(video_path, frame_start, 0, (width, height))            
                conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces, 0)
                item['conversations'].append({'from': 'human', 'value': conv_user})
                item['conversations'].append({'from': 'gpt', 'value': conv_gpt})
                item['image_data'].append(image)
                return item

            neg_tracks = neg_tracks[:, :, neg_sampled_ids.bool()]
            neg_tracks_history = neg_tracks_history[:, :, neg_sampled_ids.bool()]

            image = self._get_frame(video_path, frame_start, frame_pos, (width, height))
            if image is None:
                image = self._get_frame(video_path, frame_start, 0, (width, height))            
                conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces, 0)
                item['conversations'].append({'from': 'human', 'value': conv_user})
                item['conversations'].append({'from': 'gpt', 'value': conv_gpt})
                item['image_data'].append(image)
                return item
            
            # we have two choices: a) use visual prompting and b) use textual prompting
            if self.settings['som']['format']  == "visual":
                # image = tom_prompting(self.trace, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)            
                image, pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
                    som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=True, draw_som_negative=True)
                mark_ids = sorted([key for key in pos_traces_to_mark.keys()] + [key for key in neg_traces_to_mark.keys()])
            else:
                # image = tom_prompting(self.trace, image, pos_tracks_history, neg_tracks_history, draw_som_positive=False, draw_som_negative=False)            
                image, pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
                    som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=False, draw_som_negative=False)         
                # aggregate the starting points of the traces from pos_trace_to_mark and neg_trace_to_mark
                traces_to_mark = {**pos_traces_to_mark, **neg_traces_to_mark}                
                traces_to_mark = dict(sorted(traces_to_mark.items()))
                mark_positions = {key: (self.spatial_quant_size*val[0][0]/torch.tensor([width, height])).int().tolist() for key, val in traces_to_mark.items()}
                # turn mark_positions to str
                # mark_ids = ', '.join([f"Mark {key} at [{float(val[0])/self.spatial_quant_size:.2f},{float(val[1])/self.spatial_quant_size:.2f}]" for key, val in mark_positions.items()])
                mark_ids = ', '.join([f"Mark {key} at {val}" for key, val in mark_positions.items()])

            # visualize the traces
            if self.show_trace:
                import pdb; pdb.set_trace()
                images = [image] * pos_tracks.shape[1]
                video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
                self.trace.visualizer.save_dir = "./release/robotics"
                _ = self.trace.visualize(video, pos_tracks, pos_visibility, filename=f"{item['trace'].replace('/', '_').replace('.pth', '')}", mode="rainbow")

            pos_traces_to_mark = dict(sorted(pos_traces_to_mark.items()))
        
            mark_trace_history = ''
            mark_trace_future = ''

            valid_marks = {}
            speeds = {}
            for key, val in pos_traces_to_mark.items():
                # random select a frame position but not the last frame
                # frame_pos = torch.randint(0, trace.size(0)-1, (1,)).item()
                trace = val[0]
                trace[:, 0] = self.spatial_quant_size * trace[:, 0] / width
                trace[:, 1] = self.spatial_quant_size * trace[:, 1] / height

                trace_temp = trace.clone()
                # remove (almost) static points
                trace_temp = self.trace.remove_close_points_tensor(trace_temp, 2)
                # remove invisible points
                trace_temp = trace_temp[(trace_temp > 0).sum(1) == 2]
                # if trace_temp.size(0) <= step_to_predict // 4:
                #     continue
                # calulate motion speed
                # if trace_temp.size(0) < step_to_predict:
                #     trace_temp = torch.cat([trace_temp, trace_temp[-1].repeat(step_to_predict - trace_temp.size(0), 1)], dim=0)
                # elif trace_temp.size(0) > step_to_predict:
                #     trace_temp = trace_temp[:step_to_predict]   

                # calcualte speed
                speed = torch.norm(trace_temp[1:] - trace_temp[:-1], dim=1).mean()
                if torch.isnan(speed):
                    continue

                speeds[key] = speed.item()
                
                if speed < self.settings['trace_processor']['postive_speed_threshold']:
                    continue                
                # trace_history = trace[0]
                # val_str_history = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in trace_history.tolist()]) + ']'
                # mark_trace_history += f'\"Mark {key}\": \"{val_str_history}\"\n'    
                # round trace_temp
                if self.remove_static_trace_pts:
                    valid_marks[key] = trace_temp.int()
                else:
                    valid_marks[key] = trace.int()

                # NOTE: there was a bug here
                # val_str_future = '[' + ','.join([f'[{float(x[0])/self.spatial_quant_size:.2f},{float(x[1])/self.spatial_quant_size:.2f}]' for x in valid_marks[key][1:].tolist()]) + ']'
                val_str_future = '[' + ','.join([f'[{x[0]},{x[1]}]' for x in valid_marks[key][1:].tolist()]) + ']'

                mark_trace_future += f'\"Mark {key}\": \"{val_str_future}\"\n\n'      
            
            if len(mark_trace_future) > 0:

                num_future_steps = [val.shape[0]-1 for val in valid_marks.values()]
                step_to_predict = max(num_future_steps)                

                if self.mm_use_trace_speed:
                    # calculate the average speed of the marks
                    avg_speed = int(sum(speeds.values()) / len(speeds))
                    conv_user = (
                        f"{self.image_placeholder}\nThe image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks: {mark_ids}.\n"
                        f"The robot is doing: {gpt_response}. To finish the task, how to move the numerical marks in the image with speed {avg_speed} for the next {step_to_predict} steps?\n"
                    )
                else:
                    conv_user = (
                        f"{self.image_placeholder}\nThe image is split into {self.spatial_quant_size}x{self.spatial_quant_size} grids, and labeled with numeric marks: {mark_ids}.\n"
                        f"The robot is doing: {gpt_response}. To finish the task, how to move the numerical marks in the image for the next {step_to_predict} steps?\n"
                    )

                # formmated_val = ', '.join([f"Mark {key} at [{float(val[0][0].item())/self.spatial_quant_size:.2f},{float(val[0][1].item())/self.spatial_quant_size:.2f}]" for key, val in valid_marks.items()])       
                formmated_val = ', '.join([f"Mark {key} at [{val[0][0].item()},{val[0][1].item()}]" for key, val in valid_marks.items()])       
                if self.mm_use_trace_start_end:
                    mark_trace_future = f'<trace_start>{mark_trace_future}<trace_end>'                      
                conv_gpt = f"{formmated_val} should be moved, and their future positions are:\n\n{mark_trace_future}"
                
                item['conversations'].append({'from': 'human', 'value': conv_user})
                item['conversations'].append({'from': 'gpt', 'value': conv_gpt})
                item['image_data'].append(image)    
            else:
                for key, val in neg_traces_to_mark.items():
                    trace = val[0]
                    trace[:, 0] = self.spatial_quant_size * trace[:, 0] / width
                    trace[:, 1] = self.spatial_quant_size * trace[:, 1] / height

                conv_user, conv_gpt, image = self._construct_conv_som(item, image, visual_traces, frame_pos, pos_traces_to_mark, neg_traces_to_mark, normalize=False)
                item['conversations'].append({'from': 'human', 'value': conv_user})
                item['conversations'].append({'from': 'gpt', 'value': conv_gpt})
                item['image_data'].append(image) 

            import pdb; pdb.set_trace()
            return item
    
    def filter_items(self, items):
        """
        Filter invalid items
        """
        return items