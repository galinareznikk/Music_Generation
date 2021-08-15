import os
from flask import Flask, request, render_template, url_for, redirect, views
import pygame

app = Flask(__name__)
global paused
paused = False

@app.route("/", methods=["POST", "GET"])
def Home():
    if request.method == "POST":
        pygame.init()
        pygame.mixer.music.load("generated_midis/1.midi")
        return render_template('Page-2.html')
    return render_template('Home.html')


@app.route("/generate-page", methods=["POST", "GET"])
def generate():
    pygame.init()
    pygame.mixer.music.load("generated_midis/1.midi")
    return render_template('Page-2.html')

@app.route('/my-link/')
def my_link():
    print("HELLO")
    os.system("python generate.py 1 checkpoints/checkpoint_296.h5")
    return render_template('Page-2.html')


@app.route('/play_song/', methods=['POST', 'GET'])
def play_song():
    global paused
    pygame.init()
    pygame.mixer.music.load("generated_midis/1.midi")
    pygame.mixer.music.play()
    paused = True
    return render_template('Page-2.html', paused=paused)


@app.route('/pause_song/', methods=['POST'])
def pause_song():
    global paused
    if not paused:
        pygame.mixer.music.unpause()
        print("MORAN")
        paused = True
    else:
        print("MORAN-else")
        print(paused)
        pygame.mixer.music.pause()
        paused = False
    return render_template('Page-2.html', paused=paused)


@app.route('/stop_song/')
def stop_song():
    pygame.mixer.music.stop()
    return render_template('Page-2.html')


if __name__ == '__main__':
    app.run(debug=True)
