import sounddevice as sd
import numpy as np
import whisper
import requests
import subprocess
import time
from pynput import keyboard
import select

# ---------- CONFIG ----------
fs = 16000
chunk = 0.2
sd.default.device = (5, 5)
model = whisper.load_model("base")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"
#MODEL = "mistral"


# ---------- KEY STATE ----------
recording = False

def on_press(key):
    global recording
    if key == keyboard.Key.ctrl_r:
        recording = True

def on_release(key):
    global recording
    if key == keyboard.Key.ctrl_r:
        recording = False

keyboard.Listener(on_press=on_press, on_release=on_release).start()

# ---------- AUDIO ----------
def record():
    buf = []

    while not recording:
        time.sleep(0.05)

    while recording:
        audio = sd.rec(int(chunk * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        buf.append(np.squeeze(audio))

    return np.concatenate(buf) if buf else None

def transcribe(audio):
    return model.transcribe(audio, fp16=False)["text"].strip()

# ---------- LLM ----------
def generate(req, prompt):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": f"{prompt}\n{req}",
            "stream": False
        }
    )
    return r.json()["response"].strip()


# ---------- PERSISTENT SHELL ----------
shell = subprocess.Popen(
    ["bash"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# ---------- MAIN LOOP ----------
unique = 0
while True:
    unique = unique + 1
    marker = f"__CMD_DONE_{unique}__"

    print("\n")

    userinput = input("Type : ")
    userinput = "User's request : I am running Ubuntu OS. I have apt package manager. Provide a Linux terminal command that can does the following. " + userinput
    
    # is it achievable using Linux Terminal command
    achievable = generate(userinput, """your response can either be "True" or "False". 
                          if the user's request be achieved by running a Linux terminal command then you output "True" else you output "False"
                          no explaination.""")
    
    if (achievable == "True"):
        #print("it is achievable, let us try ...")
        tries = 0
        feedback = ""

        while(True):
            tries = tries + 1            
            if(tries > 10):
                break

            #print("\nTries :", tries)

            # generate command
            cmd = generate(userinput, """convert the user's request into a Linux command.
                        Output only the command. No explanation.""")

            cmd = cmd.strip().replace("`", "").replace("```", "")
            cmd = cmd.split("\n")[0].strip()
            #print(cmd)
            
            # validate command
            linuxcmd = "Linux command : " + cmd
            valid = generate(userinput + "\n" + feedback + "\n" + linuxcmd, """ 
                             if the Linux command will achieve what the user requested for then you output "True" , no explaination.
                             otherwise output why it will not work while keeping the your output below 30 words. Do not guess.
                             """)
            #print(valid)

            if(valid.startswith("True")):
                # safety filter
                danger = ["rm -rf", "shutdown", "reboot", "mkfs"]
                if any(d in cmd for d in danger):
                    print("Blocked: dangerous command")
                    continue

                # execute command
                shell.stdin.write(cmd + "\n")
                shell.stdin.write(f"echo {marker}\n")
                shell.stdin.flush()

                # read output safely
                while True:
                    ready, _, _ = select.select([shell.stdout], [], [], 2)

                    if shell.stdout in ready:
                        line = shell.stdout.readline()
                        if not line:
                            break

                        if marker in line:
                            break

                        print(line, end="")
                    else:
                        print("No output (timeout)")
                        break

                break
            else:
                pass

    else:
        print(achievable)

print(".............")