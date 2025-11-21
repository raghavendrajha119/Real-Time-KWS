# actions.py
import platform
import subprocess
import time

try:
    import pyautogui
except Exception as e:
    print("[WARN] PyAutoGUI not available:", e)
    pyautogui = None


# ----- Detect OS -----
OS = platform.system().lower()
print("Dedicated os: ", OS)

# ----- Cooldown & State -----
COOLDOWN = 2.0
last_trigger_time = {}

keyword_state = {
    "on": False,
    "off": False,
    "stop": False
}


def should_trigger(keyword):
    """Return True only if keyword is allowed to trigger (cooldown check)."""
    now = time.time()
    if keyword in last_trigger_time:
        if now - last_trigger_time[keyword] < COOLDOWN:
            return False
    last_trigger_time[keyword] = now
    return True


# ----- ACTIONS -----

def open_text_editor():
    if OS == "linux":
        try:
            subprocess.Popen(["gedit"])
        except:
            subprocess.Popen(["xdg-open", "."])
    elif OS == "windows":
        subprocess.Popen("notepad.exe")


def increase_volume():
    if OS == "linux":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"])
    elif OS == "windows":
        if pyautogui:
            for _ in range(5):
                pyautogui.press("volumeup")


def minimize_all_windows():
    if OS == "linux":
        if pyautogui:
            pyautogui.hotkey("ctrl","alt","d")
    elif OS == "windows":
        if pyautogui:
            pyautogui.hotkey("win","m")


# ----- MAIN ACTION HANDLER -----
def handle_keyword(keyword):
    """
    Perform the action for recognized keyword ('on', 'off', 'stop').

    Handles:
    - cooldown filter
    - per-keyword state
    - execution of respective action
    """
    if not should_trigger(keyword):
        return f"[INFO] Cooldown active — ignored {keyword}"

    # Prevent repeated actions if keyword is already active
    if keyword_state.get(keyword):
        return f"[INFO] Ignored repeated '{keyword}'"

    # Reset all states
    for k in keyword_state:
        keyword_state[k] = False

    output = ""

    if keyword == "on":
        output += "+#@+#@+#@+ Welcome +#@+#@+#@+\n"
        output += f"[ACTION] ({OS}) Opening editor + Increasing Volume"
        open_text_editor()
        increase_volume()
        keyword_state["on"] = True

    elif keyword == "off":
        output += "+#@+#@+#@+ Thank You +#@+#@+#@+\n"
        output += f"[ACTION] ({OS}) Minimizing All Windows"
        minimize_all_windows()
        keyword_state["off"] = True

    elif keyword == "stop":
        output += "+#@+#@+#@+ Have a Nice Day +#@+#@+#@+\n"
        output += f"[ACTION] ({OS}) Stopping KWS"
        keyword_state["stop"] = True
        # STOP_FLAG will be set from main file

    return output
