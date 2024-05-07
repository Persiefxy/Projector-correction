import cv2
import screeninfo

def display_image_fullscreen(image_path, monitor_id):
    # Load the image
    image = cv2.imread(image_path)

    # Get screen information
    screens = screeninfo.get_monitors()


    # Get the resolution of the specified monitor
    target_screen = screens[monitor_id]
    width, height = target_screen.width,target_screen.height
    monitors = screeninfo.get_monitors()

    # Resize the image to fit the specified monitor's resolution
    resized_image = cv2.resize(image, (width, height))

    # Create a window for the specified monitor
    cv2.namedWindow(f'Screen {monitor_id}')

    # Move the window to the specified monitor
    cv2.moveWindow(f'Screen {monitor_id}', target_screen.x-1, target_screen.y-1)
    cv2.moveWindow(f'Screen {monitor_id}', target_screen.x-1, target_screen.y-1)

    # Display the resized image full screen on the specified monitor
    cv2.setWindowProperty(f'Screen {monitor_id}', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(f'Screen {monitor_id}', resized_image)

if __name__ == '__main__':
    # Example usage
    image_path = r'C:\Users\Administrator\Desktop\muilt-Projector-correction\result\0.png'
    monitor_id = 2  # Replace with the actual monitor ID you want to display on

    display_image_fullscreen(image_path, monitor_id)
    display_image_fullscreen(image_path, 0)

    # Wait for any key press to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()