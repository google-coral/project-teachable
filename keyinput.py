import sys
import threading
import queue
import tty
import termios
import signal
import atexit

# We change the TTY settings to emit single characters (not by line) and
# ensure the settings are reversed at shutdown
old_tty_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())
def reset_tty():
  print("Reset TTY settings.")
  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_settings)
atexit.register(reset_tty)

# Make sure we dont swallow Ctrl+C
def signal_handler(sig, frame): sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)

def monitor_stdin(char_queue):
  while True:
    ch = sys.stdin.read(1) # Blocks until next char
    if ch:
      char_queue.put(ch)
      print("Key pressed:", ch)

char_queue = queue.Queue()
input_thread = threading.Thread(target=monitor_stdin, args=(char_queue,))
input_thread.daemon = True
input_thread.start()

def has_char():
  return not char_queue.empty()

def get_char():
  return None if char_queue.empty() else char_queue.get()
