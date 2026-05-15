import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import torch
import pickle
import os

# --- Core Logic Modules ---
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from view_transformer import ViewTransformer

# --- Tactical Visualization Modules ---
from pitch_config import SoccerPitchConfiguration
from radar_utils import draw_pitch, draw_points_on_pitch

def main():
    # 1. Hardware Optimization
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda:0' else 'CPU'})")

    # 2. Load Video
    # 08fd33_4.mp4 is not mandatory, you can replace it with any video of your choice. Just make sure to update the path accordingly.
    input_video = "input_videos/08fd33_4.mp4"
    print(f"Loading video frames from {input_video}...")
    video_frames = read_video(input_video)

    # 3. Model Initialization
    print("Initializing models and configuration...")
    tracker = Tracker('models/best1.pt')
    tracker.model.to(device)
    field_model = YOLO('models/best2.pt').to(device)
    pitch_config = SoccerPitchConfiguration()

    # 4. Object Tracking & Interpolation
    print("Fetching object tracks...")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks_stub.pkl")
    
    # Interpolate ball BEFORE adding positions and transforming
    print("Interpolating ball positions...")
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    tracker.add_position_to_tracks(tracks)
    
    # 5. Camera Movement
    print("Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # 6. Perspective Matrices
    print("Calculating/Loading dynamic perspective matrices...")
    matrices_stub = "stubs/homography_matrices.pkl"
    if os.path.exists(matrices_stub) and os.path.getsize(matrices_stub) > 0:
        with open(matrices_stub, 'rb') as f:
            matrices = pickle.load(f)
        print("Loaded homography matrices from stub.")
    else:
        print("Calculating matrices (this will only happen once)...")
        matrices = []
        for frame in video_frames:
            res = field_model(frame, conf=0.3)[0]
            kp = sv.KeyPoints.from_ultralytics(res)
            mask = kp.confidence[0] > 0.5
            f_pts = kp.xy[0][mask]
            p_pts = np.array(pitch_config.vertices)[mask]
            m = cv2.findHomography(f_pts, p_pts)[0] if len(f_pts) >= 4 else None
            matrices.append(m)
        with open(matrices_stub, 'wb') as f:
            pickle.dump(matrices, f)
        print("Saved homography matrices to stub.")

    # 7. Transform to 2D Metric Space
    view_transformer = ViewTransformer(pitch_config.vertices)
    view_transformer.add_transformed_positions_to_tracks(tracks, matrices)
    
    # 8. Speed Processing
    print("Processing movement metrics...")
    speed_estimator = SpeedAndDistanceEstimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # 9. Team Assignment 
    print("Assigning team colors and ball possession...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for p_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], p_id)
            
            tracks['players'][frame_num][p_id]['team'] = team
            tracks['players'][frame_num][p_id]['team_color'] = team_assigner.team_colors[team]

    # 10. Possession & Annotations (RAM OPTIMIZED) for NVIDIA graphics card users
    player_assigner = PlayerBallAssigner()
    team_control = []
    for frame_num, p_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox'] if 1 in tracks['ball'][frame_num] else None
        
        if ball_bbox is not None:
            assigned = player_assigner.assign_ball_to_players(p_track, ball_bbox)
        else:
            assigned = -1

        if assigned != -1:
            tracks['players'][frame_num][assigned]['has_ball'] = True
            team_control.append(tracks['players'][frame_num][assigned]['team'])
        else:
            team_control.append(team_control[-1] if team_control else 1)
    
    print("Applying broadcast annotations (Overwriting in RAM to save memory)...")
    
    # Overwrite the `video_frames` variable instead of creating a new list to save RAM
    video_frames = tracker.draw_annotations(video_frames, tracks, np.array(team_control))
    video_frames = camera_movement_estimator.draw_camera_movement(video_frames, camera_movement_per_frame)
    video_frames = speed_estimator.draw_speed_and_distance(video_frames, tracks)

    # 11. Tactical Radar Stitching
    print("Stitching tactical radar...")
    h, w = video_frames[0].shape[:2]
    dummy = draw_pitch(pitch_config, background_color=sv.Color.from_hex('228B22'))
    new_rw = int(h * (dummy.shape[1] / dummy.shape[0]))
    
    out_path = "output_videos/final_analysis.avi"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 24.0, (w + new_rw, h))

    for f_idx in range(len(video_frames)):
        radar = draw_pitch(pitch_config, background_color=sv.Color.from_hex('228B22'), line_color=sv.Color.WHITE)
        
        t1_pts, t2_pts, ref_pts, ball_pts = [], [], [], []
        
        # Gather Players
        for p_id, info in tracks['players'][f_idx].items():
            pos = info.get('position_transformed')
            if pos:
                if info.get('team') == 1: t1_pts.append(pos)
                else: t2_pts.append(pos)
        
        # Gather Referees
        if 'referees' in tracks:
            for r_id, info in tracks['referees'][f_idx].items():
                pos = info.get('position_transformed')
                if pos: ref_pts.append(pos)
        
        # Gather Ball
        if 'ball' in tracks:
            for b_id, info in tracks['ball'][f_idx].items():
                pos = info.get('position_transformed')
                if pos: ball_pts.append(pos)

        # Plot Points to Radar
        draw_points_on_pitch(pitch_config, np.array(t1_pts), sv.Color.from_hex('00BFFF'), sv.Color.WHITE, 12, 2, radar) # Cyan
        draw_points_on_pitch(pitch_config, np.array(t2_pts), sv.Color.from_hex('FF1493'), sv.Color.WHITE, 12, 2, radar) # Pink
        
        if ref_pts: 
            draw_points_on_pitch(pitch_config, np.array(ref_pts), sv.Color.YELLOW, sv.Color.BLACK, 12, 2, radar)
            
        if ball_pts: 
            draw_points_on_pitch(pitch_config, np.array(ball_pts), sv.Color.BLACK, sv.Color.WHITE, 8, 2, radar)

        radar_res = cv2.resize(radar, (new_rw, h))
        
        # Stitch using the frame from our RAM-optimized list
        writer.write(cv2.hconcat([video_frames[f_idx], radar_res]))
        
        # Delete the frame from memory immediately after it is written to disk
        video_frames[f_idx] = None 

    writer.release()
    print(f"Success! Saved to {out_path}")

if __name__ == "__main__":
    main()