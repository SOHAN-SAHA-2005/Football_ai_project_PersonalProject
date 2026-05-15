import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, pitch_vertices):
        """
        Initializes the transformer with target vertices (the real-world coordinates).
        """
        # Convert the list of 32 keypoints from pitch_config into a numpy array
        self.target_vertices = np.array(pitch_vertices, dtype=np.float32)

    def transform_point(self, point, matrix):
        """
        Transforms a single pixel coordinate (x, y) into a 2D metric coordinate (meters).
        """
        if matrix is None:
            return None
        
        # Reshape point for OpenCV perspectiveTransform: must be (1, 1, 2)
        reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        
        # Apply the homography matrix
        transformed = cv2.perspectiveTransform(reshaped_point, matrix)
        
        # Return as a simple [x, y] list
        return transformed.squeeze().tolist()
        
    def add_transformed_positions_to_tracks(self, tracks, matrices):
        """
        Loops through all object tracks and applies the specific homography matrix 
        calculated for each frame to get real-world coordinates.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                # Ensure we have a valid matrix for this frame
                matrix = matrices[frame_num]
                
                for track_id, track_info in track.items():
                    # Prioritize camera-adjusted positions for better accuracy
                    position = track_info.get('position_adjusted', track_info.get('position'))
                    
                    if position is not None and matrix is not None:
                        position_transformed = self.transform_point(position, matrix)
                        
                        if position_transformed is not None:
                            # Save the new real-world metric position into the tracks dictionary
                            tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed