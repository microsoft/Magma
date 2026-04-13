import faiss
import torch
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
from matplotlib import cm


class visual_tracker():
    def __init__(
            self,
            grid_size=10, 
            grid_query_frame=0, 
            backward_tracking=False, 
            save_dir="./", 
            ckpt_path=None, 
            device='cuda'
        ):

        self.grid_size = grid_size
        self.grid_query_frame = grid_query_frame
        self.backward_tracking = backward_tracking
        self.device = device

        print("Default device: ", device)

        cotracker_checkpoint = ckpt_path
        if cotracker_checkpoint is not None:
            model = CoTrackerPredictor(checkpoint=cotracker_checkpoint).to(device)
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        self.model = model
        self.visualizer = Visualizer(save_dir=save_dir, pad_value=0, linewidth=1, tracks_leave_trace=-1)

    def extract_visual_trace(self, video):
        video = video.to(self.device)
        pred_tracks, pred_visibility = self.model(
            video,
            grid_size=self.grid_size,
            # grid_query_frame=self.grid_query_frame,
            # backward_tracking=self.backward_tracking,
            # segm_mask=segm_mask
        )
        return video, pred_tracks, pred_visibility

    # def visual_trace_length(self, pred_tracks, image_size):
    #     """
    #     Compute the length of the visual trace
    #     pred_tracks: e.g., [1, 77, 225, 2]
    #     """
    #     distance_accum = 0
    #     for i in range(1, pred_tracks.size(1)):
    #         curr_pts = pred_tracks[0, i]
    #         prev_pts = pred_tracks[0, i - 1]
    #         delta = curr_pts - prev_pts
    #         distance = torch.norm(delta, dim=1)
    #         distance_accum += distance
    #     distance_accum = distance_accum * 640 / image_size[0] 
    #     return distance_accum / pred_tracks.size(1)

    def visual_trace_length(self, pred_tracks, pred_visibility, image_size):
        """
        Compute the length of the visual trace
        pred_tracks: e.g., [1, 77, 225, 2]
        pred_visibility: e.g., [1, 77, 225]
        image_size: e.g., [720, 1280]            
        """
        pred_tracks_normalized = pred_tracks / torch.tensor(image_size).float()[None, None, None, :].to(pred_tracks.device)
        pred_visiblity_float = pred_visibility[:, 1:].float().to(pred_tracks.device)
        consecutive_displacement = torch.norm(pred_tracks_normalized[:, 1:] - pred_tracks_normalized[:, :-1], dim=3)
        # average_displacement = (consecutive_displacement * pred_visiblity_float).sum(1) / (1e-5 + pred_visiblity_float.sum(1))
        average_displacement = consecutive_displacement.mean(1)
        return average_displacement    

    @classmethod
    def cluster_traces(self, traces, n_clusters=3):
        try:
            traces_for_clustering = traces[0].transpose(0, 1)
            # pred_tracks_4_clustering = pred_tracks_4_clustering - pred_tracks_4_clustering[:, :1]
            traces_for_clustering = traces_for_clustering.flatten(1)
            kmeans = faiss.Kmeans(
                traces_for_clustering.shape[1], 
                min(n_clusters, traces_for_clustering.shape[0]), 
                niter=50, 
                verbose=False,
                min_points_per_centroid=1,
                max_points_per_centroid=10000000,
            )
            kmeans.train(traces_for_clustering.cpu().numpy())
            distances, cluster_ids_x_np = kmeans.index.search(traces_for_clustering.cpu().numpy(), 1)
            cluster_ids_x = torch.from_numpy(cluster_ids_x_np).to(traces_for_clustering.device)
        except:
            print("kmeans failed")
            return None
        # sample 20% of ids or at lest 1 and at most 2 ids from each cluster
        sampled_ids = cluster_ids_x.new_zeros(cluster_ids_x.size(0)).to(traces.device)
        for cluster_id in range(min(n_clusters, traces_for_clustering.shape[0])):
            cluster_idx = (cluster_ids_x == cluster_id).nonzero().squeeze(1)                                
            num_pts_to_sample = max(1, min(1, int(0.2*cluster_idx.size(0))))
            if num_pts_to_sample > 0:
                # TODO: random sample is a bit dummy, need a better sampling algo here
                sampled_idx = torch.randperm(cluster_idx.size(0))[:num_pts_to_sample]
                sampled_ids[cluster_idx[sampled_idx]] = 1
        return sampled_ids

    def visualize(self, video, pred_tracks, pred_visibility, filename="visual_trace.mp4", mode="ranbow"):
        if mode == "rainbow":
            self.visualizer.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.visualizer.color_map = cm.get_cmap(mode)
        return self.visualizer.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if self.backward_tracking else self.grid_query_frame,
            filename=filename,
        )           