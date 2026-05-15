import cv2
import numpy as np
import supervision as sv

def draw_pitch(
    config, 
    background_color=sv.Color.from_hex('228B22'), # Forest Green
    line_color=sv.Color.WHITE, 
    padding=50, 
    scale=10.0
):
    """Draws a professional green tactical radar pitch."""
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    
    # Initialize with Grass Green
    pitch_image = np.full((scaled_width + 2 * padding, scaled_length + 2 * padding, 3), 
                         background_color.as_bgr(), dtype=np.uint8)
    
    # Draw outer boundary
    cv2.rectangle(pitch_image, (padding, padding), (scaled_length + padding, scaled_width + padding), line_color.as_bgr(), 2)
    
    # Halfway line
    halfway_x = int((scaled_length / 2) + padding)
    cv2.line(pitch_image, (halfway_x, padding), (halfway_x, scaled_width + padding), line_color.as_bgr(), 2)
    
    # Center Circle
    center_y = int((scaled_width / 2) + padding)
    cv2.circle(pitch_image, (halfway_x, center_y), int(9.15 * scale), line_color.as_bgr(), 2)
    cv2.circle(pitch_image, (halfway_x, center_y), 3, line_color.as_bgr(), -1)

    # Penalty Areas
    def draw_penalty_area(x_start, direction):
        # Large box
        cv2.rectangle(pitch_image, 
                     (x_start, int(padding + (config.width - 40.32)/2 * scale)),
                     (x_start + int(direction * 16.5 * scale), int(padding + (config.width + 40.32)/2 * scale)),
                     line_color.as_bgr(), 2)
        # Small box
        cv2.rectangle(pitch_image, 
                     (x_start, int(padding + (config.width - 18.32)/2 * scale)),
                     (x_start + int(direction * 5.5 * scale), int(padding + (config.width + 18.32)/2 * scale)),
                     line_color.as_bgr(), 2)

    draw_penalty_area(padding, 1) # Left
    draw_penalty_area(scaled_length + padding, -1) # Right
    
    return pitch_image

def draw_pitch_voronoi_diagram(
    config,
    team_1_xy,
    team_2_xy,
    team_1_color=sv.Color.from_hex('00BFFF'), # Cyan
    team_2_color=sv.Color.from_hex('FF1493'), # Pink
    opacity=0.3, 
    padding=50,
    scale=10.0,
    pitch=None
):
    """Calculates Voronoi regions and blends them onto the pitch."""
    calc_scale = 3.0 # Lower scale for faster math
    scaled_w = int(config.width * calc_scale)
    scaled_l = int(config.length * calc_scale)
    
    voronoi_layer = np.zeros((scaled_w + 2 * padding, scaled_l + 2 * padding, 3), dtype=np.uint8)

    t1_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    t2_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y, x = np.indices((voronoi_layer.shape[0], voronoi_layer.shape[1]))
    y_norm, x_norm = y - padding, x - padding

    def get_dists(xy):
        if len(xy) == 0: return np.full_like(x_norm, 1e6, dtype=np.float32)
        return np.sqrt((xy[:, 0][:, None, None] * calc_scale - x_norm) ** 2 +
                       (xy[:, 1][:, None, None] * calc_scale - y_norm) ** 2)

    d1, d2 = np.min(get_dists(team_1_xy), axis=0), np.min(get_dists(team_2_xy), axis=0)
    blend = np.tanh((d2 / np.clip(d1 + d2, 1e-5, None) - 0.5) * 15) * 0.5 + 0.5

    for c in range(3):
        voronoi_layer[:, :, c] = (blend * t1_bgr[c] + (1 - blend) * t2_bgr[c]).astype(np.uint8)

    voronoi_layer = cv2.resize(voronoi_layer, (pitch.shape[1], pitch.shape[0]))
    return cv2.addWeighted(voronoi_layer, opacity, pitch, 1 - opacity, 0)

def draw_points_on_pitch(
    config, 
    xy, 
    face_color, 
    edge_color, 
    radius, 
    thickness, 
    pitch, 
    scale=10.0, 
    padding=50
):
    """Plots individual player or ball coordinates on the radar map."""
    for point in xy:
        x = int(point[0] * scale) + padding
        y = int(point[1] * scale) + padding
        
        # Fill circle
        cv2.circle(pitch, (x, y), radius, face_color.as_bgr(), -1)
        # Draw stroke/outline
        cv2.circle(pitch, (x, y), radius, edge_color.as_bgr(), thickness)
        
    return pitch