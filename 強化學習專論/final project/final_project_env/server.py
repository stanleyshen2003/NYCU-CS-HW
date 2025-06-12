import argparse
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, request, render_template, send_file

from racecar_gym.env import RaceEnv

#
SERVER_RAISE_EXCEPTION = True

# Time unit: second
MAX_ACCU_TIME = -1

#
env, reward, trunc, info = None, None, None, None
obs: Optional[np.array] = None
sid: Optional[str] = None
output_freq: Optional[int] = None
terminal = False
images = []
step = 0
port: Optional[int] = None
host: Optional[str] = None
scenario: Optional[str] = None

#
app = Flask(__name__)

#
accu_time = 0.
last_get_obs_time: Optional[float] = None
end_time: Optional[float] = None


def get_img_views():
    global obs, sid, info, env

    progress = info['progress']
    lap = int(info['lap'])
    score = lap + progress - 1.

    # Get the images
    img1 = env.env.force_render(render_mode='rgb_array_higher_birds_eye', width=540, height=540,
                                position=np.array([4.89, -9.30, -3.42]), fov=120)
    img2 = env.env.force_render(render_mode='rgb_array_birds_eye', width=270, height=270)
    img3 = env.env.force_render(render_mode='rgb_array_follow', width=128, height=128)
    img4 = (obs.transpose((1, 2, 0))).astype(np.uint8)

    # Combine the images
    img = np.zeros((540, 810, 3), dtype=np.uint8)
    img[0:540, 0:540, :] = img1
    img[:270, 540:810, :] = img2
    img[270 + 10:270 + 128 + 10, 540 + 7:540 + 128 + 7, :] = img3
    img[270 + 10:270 + 128 + 10, 540 + 128 + 14:540 + 128 + 128 + 14, :] = img4

    # Draw the text
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('./racecar_gym/Arial.ttf', 25)
    font_large = ImageFont.truetype('./racecar_gym/Arial.ttf', 35)
    draw.text((5, 5), "Full Map", font=font, fill=(255, 87, 34))
    draw.text((550, 10), "Bird's Eye", font=font, fill=(255, 87, 34))
    draw.text((550, 280), "Follow", font=font, fill=(255, 87, 34))
    draw.text((688, 280), "Obs", font=font, fill=(255, 87, 34))
    draw.text((550, 408), f"Lap {lap}", font=font, fill=(255, 255, 255))
    draw.text((688, 408), f"Prog {progress:.3f}", font=font, fill=(255, 255, 255))
    draw.text((550, 450), f"Score {score:.3f}", font=font_large, fill=(255, 255, 255))
    draw.text((550, 500), f"ID {sid}", font=font_large, fill=(255, 255, 255))

    img = np.asarray(img)

    return img


def record_video(filename: str):
    global images
    height, width, layers = images[0].shape
    # noinspection PyUnresolvedReferences
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, 30, (width, height))
    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def get_observation(port_num):
    """Return the 3x128x128"""
    try:
        global obs, last_get_obs_time

        if terminal:
            return jsonify({'terminal': terminal})

        # Record time
        last_get_obs_time = time.time()

        return jsonify({'observation': obs.tolist()})
    except Exception as e:
        if SERVER_RAISE_EXCEPTION:
            raise e
        print(e)
        return jsonify({'error': str(e)})


def set_action(port_num):
    try:
        global obs, reward, terminal, trunc, info, step, output_freq, sid, accu_time

        if terminal:
            return jsonify({'terminal': terminal})

        action = request.json.get('action')

        accu_time += time.time() - last_get_obs_time

        step += 1
        obs, _, terminal, trunc, info = env.step(action)

        progress = info['progress']
        lap = int(info['lap'])
        score = lap + progress - 1.
        env_time = info['time']

        # Print information
        print_info = f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step: {step} Lap: {info["lap"]}, ' \
                     f'Progress: {info["progress"]:.3f}, ' \
                     f'EnvTime: {info["time"]:.3f} ' \
                     f'AccTime: {accu_time:.3f} '
        if info.get('n_collision') is not None:
            print_info += f'Collision: {info["n_collision"]} '
        if info.get('collision_penalties') is not None:
            print_info += 'CollisionPenalties: '
            for penalty in info['collision_penalties']:
                print_info += f'{penalty:.3f} '

        print(print_info)

        # plt.imshow(obs.transpose(1, 2, 0))
        # plt.show()

        if step % output_freq == 0:
            img = get_img_views()
            # plt.imshow(img)
            # plt.show()
            images.append(img)

        if terminal:
            if round(accu_time) > MAX_ACCU_TIME:
                print(f'[Time Limit Error] Accu time "{accu_time}" violate the limit {MAX_ACCU_TIME} (sec)!')
            cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            video_name = f'results/{sid}_{cur_time}_env{env_time:.3f}_acc{round(accu_time)}s_score{score:.4f}.mp4'
            Path(video_name).parent.mkdir(parents=True, exist_ok=True)
            record_video(video_name)
            print(f'============ Terminal ============')
            print(f'Video saved to {video_name}!')
            print(f'===================================')

        return jsonify({'terminal': bool(terminal)})
    except Exception as e:
        if SERVER_RAISE_EXCEPTION:
            raise e
        print(e)
        return jsonify({'error': str(e)})


@app.route('/<int:port_num>', methods=['GET'])
def get_obs_with_port(port_num):
    return get_observation(port_num)


@app.route('/', methods=['GET'])
def get_obs_without_port():
    return get_observation(-1)


@app.route('/<int:port_num>', methods=['POST'])
def set_action_with_port(port_num):
    return set_action(port_num)


@app.route('/', methods=['POST'])
def set_action_without_port():
    return set_action(-1)


@app.route('/realtime', methods=['GET'])
def realtime():
    return render_template('realtime_eval.html')


@app.route('/get_img')
def get_obs():
    global obs
    img = Image.fromarray(obs.transpose((1, 2, 0)))

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(
        img_byte_arr,
        mimetype='image/png'
    )


def get_args():
    global sid, output_freq, port, scenario, host, MAX_ACCU_TIME
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-freq', type=int, default=5, help='output frequency')
    parser.add_argument('--sid', type=str, required=True, help='The id of the student.')
    parser.add_argument('--port', type=int, default=5000, help='The port of the server.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host of the server.')
    parser.add_argument('--scenario', type=str, required=True, help='The scenario name.')
    args = parser.parse_args()
    sid = args.sid
    output_freq = args.output_freq
    port = args.port
    host = args.host
    scenario = args.scenario
    if 'austria' in scenario:
        MAX_ACCU_TIME = 900
    else:
        MAX_ACCU_TIME = 600


if __name__ == '__main__':
    get_args()

    env = RaceEnv(scenario=scenario,
                  render_mode='rgb_array_birds_eye',
                  reset_when_collision=True if 'austria' in scenario else False)
    obs, info = env.reset()
    #
    app.run(debug=False, host='0.0.0.0', port=port)
