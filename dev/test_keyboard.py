from pynput import keyboard

print("Press F10...")

def on_press(key):
    if key == keyboard.Key.f10:
        print("F10 PRESSED!")

def on_release(key):
    if key == keyboard.Key.f10:
        print("F10 RELEASED!")
        return False

with keyboard.Listener(
    on_press=on_press,
    on_release=on_release
) as listener:
    listener.join()