import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import io
import math
from streamlit_drawable_canvas import st_canvas
import pandas as pd


def main():
    st.set_page_config(page_title="Room Dimension Calculator", layout="wide")
    st.title("Room Dimension Calculator")

    # Initialize session state variables
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'calibration_points' not in st.session_state:
        st.session_state.calibration_points = []
    if 'scale' not in st.session_state:
        st.session_state.scale = None
    if 'current_room' not in st.session_state:
        st.session_state.current_room = []
    if 'rooms' not in st.session_state:
        st.session_state.rooms = []
    if 'mode' not in st.session_state:
        st.session_state.mode = 'upload'
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'last_click' not in st.session_state:
        st.session_state.last_click = None

    # Sidebar with instructions and controls
    with st.sidebar:
        st.header("Instructions")

        if st.session_state.mode == 'upload':
            st.info("📤 Upload a floor plan image to start")

        elif st.session_state.mode == 'calibrate':
            st.info("📏 Click on two points with a known distance between them")
            st.write(f"Points selected: {len(st.session_state.calibration_points)}/2")

            if len(st.session_state.calibration_points) == 2:
                st.number_input("Enter known distance (meters)",
                                key="known_distance",
                                min_value=0.1,
                                step=0.1,
                                help="This is the actual distance between the two points you selected")

                if st.button("Set Scale", key="set_scale"):
                    # Calculate scale (pixels per meter)
                    p1 = st.session_state.calibration_points[0]
                    p2 = st.session_state.calibration_points[1]
                    pixel_distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                    st.session_state.scale = pixel_distance / st.session_state.known_distance
                    st.session_state.mode = 'mark'
                    st.session_state.canvas_key += 1  # Force canvas refresh
                    st.experimental_rerun()

        elif st.session_state.mode == 'mark':
            st.info("🏠 Click on corners of a room in sequence")
            st.write(f"Corner points: {len(st.session_state.current_room)}")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Complete Room", disabled=len(st.session_state.current_room) < 3):
                    complete_current_room()
                    st.session_state.canvas_key += 1  # Force canvas refresh
                    st.experimental_rerun()
            with col_b:
                if st.button("Clear Points"):
                    st.session_state.current_room = []
                    st.session_state.canvas_key += 1  # Force canvas refresh
                    st.experimental_rerun()

        # Reset button always available
        if st.button("Reset Everything", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

    # Main content area
    # File upload (if no image loaded yet)
    if st.session_state.mode == 'upload':
        uploaded_file = st.file_uploader("Upload your floor plan image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Load and store the image
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.image = image
            st.session_state.mode = 'calibrate'
            st.experimental_rerun()

    # If we have an image, show it with the drawable canvas
    if st.session_state.image is not None:
        # Create columns for layout
        col1, col2 = st.columns([3, 1])

        with col1:
            # Process and create the display image with existing points/rooms
            display_image = process_image_for_display()

            # Create the canvas for drawing
            h, w = display_image.size

            # Determine canvas mode based on application mode
            if st.session_state.mode == 'calibrate':
                drawing_mode = "point"
                stroke_color = "red"
                realtime_update = True
            elif st.session_state.mode == 'mark':
                drawing_mode = "point"
                stroke_color = "blue"
                realtime_update = True

            # Display the canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color=stroke_color,
                background_image=display_image,
                update_streamlit=True,
                height=min(700, display_image.height),
                width=min(1000, display_image.width),
                drawing_mode=drawing_mode,
                key=f"canvas_{st.session_state.canvas_key}",
                point_display_radius=3,
            )

            # Handle canvas click events
            if canvas_result.json_data is not None:
                objects = pd.json_normalize(canvas_result.json_data["objects"])

                # Process clicks
                if not objects.empty and len(objects) > 0:
                    # Get the last added point
                    if "left" in objects.columns and "top" in objects.columns:
                        last_point = (int(objects.iloc[-1]["left"]), int(objects.iloc[-1]["top"]))

                        # Check if this is a new point (avoid double processing)
                        if st.session_state.last_click != last_point:
                            st.session_state.last_click = last_point

                            # Add the point to the appropriate collection
                            if st.session_state.mode == 'calibrate' and len(st.session_state.calibration_points) < 2:
                                st.session_state.calibration_points.append(last_point)
                                st.experimental_rerun()
                            elif st.session_state.mode == 'mark':
                                st.session_state.current_room.append(last_point)
                                st.experimental_rerun()

        with col2:
            # Display measured rooms
            if st.session_state.rooms:
                st.subheader("Measured Rooms")
                for i, room in enumerate(st.session_state.rooms):
                    with st.expander(f"Room {i + 1}", expanded=i == len(st.session_state.rooms) - 1):
                        st.write(f"Width: {room['width']:.2f} m")
                        st.write(f"Length: {room['length']:.2f} m")
                        st.write(f"Area: {room['area']:.2f} m²")

                        # Option to delete this room
                        if st.button(f"Delete Room {i + 1}", key=f"del_room_{i}"):
                            st.session_state.rooms.pop(i)
                            st.session_state.canvas_key += 1  # Force canvas refresh
                            st.experimental_rerun()


def process_image_for_display():
    """Create a display image with all annotations."""
    if isinstance(st.session_state.image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(st.session_state.image)
    else:
        # Clone the PIL Image
        image = st.session_state.image.copy()

    draw = ImageDraw.Draw(image)

    # Draw calibration points and line
    if st.session_state.calibration_points:
        for i, point in enumerate(st.session_state.calibration_points):
            # Draw point as a circle
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red')
            # Add label
            draw.text((point[0] + 8, point[1] - 10), f"{i + 1}", fill='red')

        # Draw line between calibration points if there are two
        if len(st.session_state.calibration_points) == 2:
            p1 = st.session_state.calibration_points[0]
            p2 = st.session_state.calibration_points[1]
            draw.line([p1, p2], fill='red', width=2)

            # Show distance if scale is available
            if st.session_state.scale:
                pixel_dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                meters = pixel_dist / st.session_state.scale
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                draw.text((mid_x, mid_y), f"{meters:.2f}m", fill='black')

    # Draw current room points and lines
    if st.session_state.current_room:
        points = st.session_state.current_room

        # Draw all points
        for i, point in enumerate(points):
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='blue')
            draw.text((point[0] + 8, point[1] - 10), str(i + 1), fill='blue')

            # Connect with lines
            if i > 0:
                draw.line([points[i - 1], point], fill='blue', width=2)

        # Close the shape if 3+ points
        if len(points) >= 3:
            draw.line([points[-1], points[0]], fill='blue', width=2)

    # Draw completed rooms
    for i, room in enumerate(st.session_state.rooms):
        points = room['points']

        # Different color for each room (cycle through some pastel colors)
        colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200),
                  (255, 200, 255), (200, 255, 255)]
        color = colors[i % len(colors)]

        # Draw polygon
        if len(points) >= 3:
            draw.polygon(points, outline='black', fill=color + (100,))  # Semi-transparent fill

            # Draw dimensions in center
            center_x = sum(p[0] for p in points) // len(points)
            center_y = sum(p[1] for p in points) // len(points)
            label = f"{room['width']:.2f}m × {room['length']:.2f}m"

            # Draw text with background for better visibility
            text_bbox = draw.textbbox((center_x, center_y), label)
            draw.rectangle([
                text_bbox[0] - 5, text_bbox[1] - 5,
                text_bbox[2] + 5, text_bbox[3] + 5
            ], fill='white')
            draw.text((center_x, center_y), label, fill='black')

    return image


def complete_current_room():
    """Calculate dimensions and add room to the list."""
    if len(st.session_state.current_room) < 3 or not st.session_state.scale:
        return

    points = np.array(st.session_state.current_room)

    # Get bounding box
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # Calculate dimensions in meters
    width = (max_x - min_x) / st.session_state.scale
    length = (max_y - min_y) / st.session_state.scale
    area = width * length

    # For irregular polygons, calculate actual area
    if len(points) > 3:
        # Calculate area using Shoelace formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2.0
        # Convert to square meters
        area = area / (st.session_state.scale ** 2)

    # Add room to list
    st.session_state.rooms.append({
        'points': st.session_state.current_room.copy(),
        'width': width,
        'length': length,
        'area': area
    })

    # Clear current room
    st.session_state.current_room = []


if __name__ == "__main__":
    main()