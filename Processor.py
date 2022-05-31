import copy
import datetime
import shutil
import os
import subprocess
import time
import re
import traceback
from typing import Dict, List, Tuple, Union
import logging
import ffmpeg
import utils
from BiliLive import BiliLive
from itertools import groupby
import jsonlines
from xml.dom.minidom import Document
from time import strftime
from time import gmtime


def write_ass(ass_start: int, ass_end: int, user: str, price: int, sctext: str):
    ass_start_str = strftime("%H:%M:%S", gmtime(ass_start))
    ass_end_str = strftime("%H:%M:%S", gmtime(ass_end))
    ncount = int(sctext.count('\\N'))
    h1 = str(961-ncount*35)
    h2 = str(1032-ncount*35)
    h3 = str(966-ncount*35)
    h4 = str(1001-ncount*35)

    h5 = str(26+ncount*35)
    h6 = str(35 + ncount * 35)
    h7 = str(43 + ncount * 35)

    s1='m 0 0 l 500 0 l 500 '+h5+' b 500 '+h6+' 492 '+h7+' 483 '+h7+' l 17 '+h7+'b 8 '+h7+' 0 '+h6+' 0 '+h5+''
    if price >= 1000:
        t1 = '{\move(-90,' + h1 + ',10,' + h1 + ')\c&HE5E5FF\shad0\p1}'
        t6 = '{\pos(10,' + h1 + ')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&HE5E5FF\shad0\p1}'
        t2 = '{\move(-90,'+h2+',10,'+h2+')\shad0\p1\c&H8C8CF7}'
        t7 = '{\pos(10,'+h2+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\shad0\p1\c&H8C8CF7}'
        t3 = '{\move(-82,'+h3+',18,'+h3+')\c&H0F0F75\\fs35\\b1\q2}'
        t8 = '{\pos(18,'+h3+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&H0F0F75\\fs35\\b1\q2}'
    elif price >= 500:
        t1 = '{\move(-90,' + h1 + ',10,' + h1 + ')\c&HD4F6FF\shad0\p1}'
        t6 = '{\pos(10,' + h1 + ')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&HD4F6FF\shad0\p1}'
        t2 = '{\move(-90,'+h2+',10,'+h2+')\shad0\p1\c&H8CCEF7}'
        t7 = '{\pos(10,'+h2+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\shad0\p1\c&H8CCEF7}'
        t3 = '{\move(-82,'+h3+',18,'+h3+')\c&H236C64\\fs35\\b1\q2}'
        t8 = '{\pos(18,'+h3+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&H236C64\\fs35\\b1\q2}'
    elif price >= 100:
        t1 = '{\move(-90,' + h1 + ',10,' + h1 + ')\c&HCAF9F8\shad0\p1}'
        t6 = '{\pos(10,' + h1 + ')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&HCAF9F8\shad0\p1}'
        t2 = '{\move(-90,'+h2+',10,'+h2+')\shad0\p1\c&H76E8E9}'
        t7 = '{\pos(10,'+h2+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\shad0\p1\c&H76E8E9}'
        t3 = '{\move(-82,'+h3+',18,'+h3+')\c&H1A8B87\\fs35\\b1\q2}'
        t8 = '{\pos(18,'+h3+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&H1A8B87\\fs35\\b1\q2}'
    else:
        t1 = '{\move(-90,' + h1 + ',10,' + h1 + ')\c&HFCE8D8\shad0\p1}'
        t6 = '{\pos(10,' + h1 + ')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&HFCE8D8\shad0\p1}'
        t2 = '{\move(-90,'+h2+',10,'+h2+')\shad0\p1\c&HE4A47A}'
        t7 = '{\pos(10,'+h2+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\shad0\p1\c&HE4A47A}'
        t3 = '{\move(-82,'+h3+',18,'+h3+')\c&H8A3619\\fs35\\b1\q2}'
        t8 = '{\pos(18,'+h3+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&H8A3619\\fs35\\b1\q2}'
    t4 = '{\move(-82,'+h4+',18,'+h4+')\c&H313131\\fs28\q2}'
    t5 = '{\move(-82,'+h2+',18,'+h2+')\c&HFFFFFF\q2}'
    t9 = '{\pos(18,'+h4+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&H313131\\fs28\q2}'
    t10 = '{\pos(18,'+h2+')\clip(m 10 17 b 10 8 18 0 27 0 l 493 0 b 502 0 510 8 510 17 l 510 1080 l 10 1080)\c&HFFFFFF\q2}'

    assstr = f'Dialogue: 0,{ass_start_str}.25,{ass_start_str}.50,message_box,,0000,0000,0000,,{t1}m 0 17 b 0 8 8 0 17 0 l 483 0 b 492 0 500 8 500 17 l 500 71 l 0 71\nDialogue: 0,{ass_start_str}.25,{ass_start_str}.50,message_box,,0000,0000,0000,,{t2}{s1}\nDialogue: 1,{ass_start_str}.25,{ass_start_str}.50,message_box,,0000,0000,0000,,{t3}{user}\nDialogue: 1,{ass_start_str}.25,{ass_start_str}.50,message_box,,0000,0000,0000,,{t4}SuperChat CNY {price}\nDialogue: 1,{ass_start_str}.25,{ass_start_str}.50,message_box,,0000,0000,0000,,{t5}{sctext}\n\nDialogue: 0,{ass_start_str}.50,{ass_end_str}.00,message_box,,0000,0000,0000,,{t6}m 0 17 b 0 8 8 0 17 0 l 483 0 b 492 0 500 8 500 17 l 500 71 l 0 71\nDialogue: 0,{ass_start_str}.50,{ass_end_str}.00,message_box,,0000,0000,0000,,{t7}{s1}\nDialogue: 1,{ass_start_str}.50,{ass_end_str}.00,message_box,,0000,0000,0000,,{t8}{user}\nDialogue: 1,{ass_start_str}.50,{ass_end_str}.00,message_box,,0000,0000,0000,,{t9}SuperChat CNY {price}\nDialogue: 1,{ass_start_str}.50,{ass_end_str}.00,message_box,,0000,0000,0000,,{t10}{sctext}\n'
    return assstr


def parse_danmu(start_time: datetime.datetime, dir_name):
    danmu_list = []
    if os.path.exists(os.path.join(dir_name, 'danmu.jsonl')):
        with jsonlines.open(os.path.join(dir_name, 'danmu.jsonl')) as reader:
            for obj in reader:
                danmu_list.append({
                    "text": obj['text'],
                    "time": obj['properties']['time'] // 1000,
                    "uid": str(obj['user_info']['user_id'])
                })
    if os.path.exists(os.path.join(dir_name, 'superchat.jsonl')):
        foc = open(os.path.join(dir_name, 'superchat.ass'), 'w', encoding='utf-8')
        foc.write(
            "[Script Info]\nScriptType: v4.00+\nCollisions: Normal\nPlayResX: 1920\nPlayResY: 1080\nTimer: 100.0000\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\nStyle: R2L,Microsoft YaHei,38,&H37FFFFFF,&H00FFFFFF,&H00000000,&H1E6A5149,0,0,0,0,100.00,100.00,0.00,0.00,1,0,1,8,0,0,0,1\nStyle: L2R,Microsoft YaHei,38,&H37FFFFFF,&H00FFFFFF,&H00000000,&H1E6A5149,0,0,0,0,100.00,100.00,0.00,0.00,1,0,1,8,0,0,0,1\nStyle: TOP,Microsoft YaHei,38,&H37FFFFFF,&H00FFFFFF,&H00000000,&H1E6A5149,0,0,0,0,100.00,100.00,0.00,0.00,1,0,1,8,0,0,0,1\nStyle: BTM,Microsoft YaHei,38,&H37FFFFFF,&H00FFFFFF,&H00000000,&H1E6A5149,0,0,0,0,100.00,100.00,0.00,0.00,1,0,1,8,0,0,0,1\nStyle: SP,Microsoft YaHei,38,&H00FFFFFF,&H00FFFFFF,&H00000000,&H1E6A5149,0,0,0,0,100.00,100.00,0.00,0.00,1,0,1,7,0,0,0,1\nStyle: message_box,Microsoft YaHei,35,&H00FFFFFF,&H00FFFFFF,&H00000000,&H1E6A5149,0,0,0,0,100.00,100.00,0.00,0.00,1,0,1,7,0,0,0,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        last_endts = 0
        with jsonlines.open(os.path.join(dir_name, 'superchat.jsonl')) as reader:
            for obj in reader:
                danmu_list.append({
                    "text": obj['text'],
                    "time": obj['time'],
                    "uid": str(obj['user_id'])
                })
                startts = int(obj['time'])
                interval = 5
                if startts <= int(start_time.timestamp()):
                    startts = int(start_time.timestamp())
                if startts <= last_endts:
                    startts = last_endts
                endts = startts + interval
                ass_start = startts - int(start_time.timestamp())
                ass_end = endts - int(start_time.timestamp())
                user = str(obj['user_name'])
                price = int(obj['price'])
                message = str(obj['text'])
                sctext = '\\N'.join([message[i:i + 17] for i in range(0, len(message), 17)])
                last_endts = endts
                foc.write(write_ass(ass_start, ass_end, user, price, sctext))
        foc.close()
    danmu_list = sorted(danmu_list, key=lambda x: x['time'])
    return danmu_list


def get_cut_points(time_dict: Dict[datetime.datetime, List[str]], up_ratio: float = 2, down_ratio: float = 0.75,
                   topK: int = 5) -> List[Tuple[datetime.datetime, datetime.datetime, List[str]]]:
    status = 0
    cut_points = []
    prev_num = None
    start_time = None
    temp_texts = []
    for time, texts in time_dict.items():
        if prev_num is None:
            start_time = time
            temp_texts = copy.copy(texts)
        elif status == 0 and len(texts) >= prev_num * up_ratio:
            status = 1
            temp_texts.extend(texts)
        elif status == 1 and len(texts) < prev_num * down_ratio:
            tags = utils.get_words("。".join(texts), topK=topK)
            cut_points.append((start_time, time, tags))
            status = 0
            start_time = time
            temp_texts = copy.copy(texts)
        elif status == 0:
            start_time = time
            temp_texts = copy.copy(texts)
        prev_num = len(texts)
    return cut_points


def get_manual_cut_points(danmu_list: List[Dict], uid: str) -> List[
    Tuple[datetime.datetime, datetime.datetime, List[str]]]:
    cut_points = []
    count = 0
    for danmu_obj in danmu_list:
        if danmu_obj['uid'] == uid and danmu_obj['text'].startswith("/DDR clip"):
            count += 1
            args = danmu_obj['text'].split()
            duration = int(args[2])
            end_time = datetime.datetime.fromtimestamp(danmu_obj['time'])
            start_time = end_time - datetime.timedelta(seconds=duration)
            hint_text = f"手动切片_{count}"
            if len(args) >= 4:
                hint_text = " ".join(args[3:])
            cut_points.append((start_time, end_time, [hint_text]))
    return cut_points


def get_true_timestamp(video_times: List[Tuple[datetime.datetime, float]], point: datetime.datetime) -> float:
    time_passed = 0
    for t, d in video_times:
        if point < t:
            return time_passed
        elif point - t <= datetime.timedelta(seconds=d):
            return time_passed + (point - t).total_seconds()
        else:
            time_passed += d
    return time_passed


def count(danmu_list: List, live_start: datetime.datetime, live_duration: float, interval: int = 60) -> Dict[
    datetime.datetime, List[str]]:
    start_timestamp = int(live_start.timestamp())
    return_dict = {}
    for k, g in groupby(danmu_list, key=lambda x: (x['time'] - start_timestamp) // interval):
        return_dict[datetime.datetime.fromtimestamp(
            k * interval + start_timestamp)] = []
        for o in list(g):
            return_dict[datetime.datetime.fromtimestamp(
                k * interval + start_timestamp)].append(o['text'])
    return return_dict


def flv2ts(input_file: str, output_file: str, ffmpeg_logfile_hander) -> Union[
    subprocess.CompletedProcess, subprocess.CalledProcessError]:
    try:
        ret = subprocess.run(
            f"ffmpeg -y -fflags +discardcorrupt -i {input_file} -c copy -bsf:v h264_mp4toannexb -acodec aac -f mpegts {output_file}",
            shell=True, check=True, stdout=ffmpeg_logfile_hander, stderr=ffmpeg_logfile_hander)
        return ret
    except subprocess.CalledProcessError as err:
        traceback.print_exc()
        return err


def xml2ass(input_file: str, output_file: str, ffmpeg_logfile_hander) -> Union[
    subprocess.CompletedProcess, subprocess.CalledProcessError]:
    try:
        ret = subprocess.run(
            f"dfcli -o {output_file} -i {input_file}", shell=True, check=True, stdout=ffmpeg_logfile_hander,
            stderr=ffmpeg_logfile_hander)
        return ret
    except subprocess.CalledProcessError as err:
        traceback.print_exc()
        return err


def concat(merge_conf_path: str, merged_file_path: str, ass_file_path: str, ffmpeg_logfile_hander) -> Union[
    subprocess.CompletedProcess, subprocess.CalledProcessError]:
    try:
        if os.path.exists(ass_file_path):
            ret = subprocess.run(
                f'ffmpeg -y -f concat -safe 0 -i {merge_conf_path} -acodec aac -ab 260k -bsf:v h264_mp4toannexb -c:v h264_nvenc -b:v 5900k -f mpegts -vf "scale=1920:1080,subtitles={ass_file_path}" -fflags +igndts -avoid_negative_ts make_zero {merged_file_path}',
                shell=True, check=True, stdout=ffmpeg_logfile_hander, stderr=ffmpeg_logfile_hander)
        else:
            ret = subprocess.run(
                f'ffmpeg -y -f concat -safe 0 -i {merge_conf_path} -acodec aac -ab 260k -bsf:v h264_mp4toannexb -c:v h264_nvenc -b:v 5900k -f mpegts -vf "scale=1920:1080" -fflags +igndts -avoid_negative_ts make_zero {merged_file_path}',
                shell=True, check=True, stdout=ffmpeg_logfile_hander, stderr=ffmpeg_logfile_hander)
        return ret
    except subprocess.CalledProcessError as err:
        traceback.print_exc()
        return err


def get_start_time(filename: str) -> datetime.datetime:
    base = os.path.splitext(filename)[0]
    return datetime.datetime.strptime(
        " ".join(base.split("_")[1:3]), '%Y-%m-%d %H-%M-%S')


class Processor(BiliLive):
    def __init__(self, config: Dict, record_dir: str, danmu_path: str):
        super().__init__(config)
        self.record_dir = record_dir
        self.danmu_path = danmu_path
        self.global_start = utils.get_global_start_from_records(
            self.record_dir)
        self.merge_conf_path = utils.get_merge_conf_path(
            self.room_id, self.global_start, config.get('root', {}).get('data_path', "./"))
        self.merged_file_path = utils.get_merged_filename(
            self.room_id, self.global_start, config.get('root', {}).get('data_path', "./"))
        self.outputs_dir = utils.init_outputs_dir(
            self.room_id, self.global_start, config.get('root', {}).get('data_path', "./"))
        self.splits_dir = utils.init_splits_dir(
            self.room_id, self.global_start, self.config.get('root', {}).get('data_path', "./"))
        self.times = []
        self.live_start = self.global_start
        self.live_duration = 0
        logging.basicConfig(level=utils.get_log_level(config),
                            format='%(asctime)s %(thread)d %(threadName)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=os.path.join(config.get('root', {}).get('logger', {}).get('log_path', "./log"),
                                                  "Processor_" + datetime.datetime.now(
                                                  ).strftime('%Y-%m-%d_%H-%M-%S') + '.log'),
                            filemode='a')
        self.ffmpeg_logfile_hander = open(
            os.path.join(config.get('root', {}).get('logger', {}).get('log_path', "./log"),
                         "FFMpeg_" + datetime.datetime.now(
                         ).strftime('%Y-%m-%d_%H-%M-%S') + '.log'), mode="a", encoding="utf-8")

    def pre_concat(self) -> Union[subprocess.CompletedProcess, subprocess.CalledProcessError]:
        filelist = os.listdir(self.record_dir)
        with open(self.merge_conf_path, "w", encoding="utf-8") as f:
            for filename in filelist:
                if os.path.splitext(
                        os.path.join(self.record_dir, filename))[1] == ".flv" and os.path.getsize(
                    os.path.join(self.record_dir, filename)) > 1024 * 1024:
                    ts_path = os.path.splitext(os.path.join(
                        self.record_dir, filename))[0] + ".ts"
                    ret = flv2ts(os.path.join(
                        self.record_dir, filename), ts_path, self.ffmpeg_logfile_hander)
                    if isinstance(ret, subprocess.CompletedProcess) and not self.config.get('spec', {}).get('recorder',
                                                                                                            {}).get(
                        'keep_raw_record', False):
                        os.remove(os.path.join(self.record_dir, filename))
                    # ts_path = os.path.join(self.record_dir, filename)
                    duration = float(ffmpeg.probe(ts_path)[
                                         'format']['duration'])
                    start_time = get_start_time(filename)
                    self.times.append((start_time, duration))
                    f.write(
                        f"file '{os.path.abspath(ts_path)}'\n")
        parse_danmu(self.global_start, self.danmu_path)
        # xml2ass(os.path.join(self.danmu_path, 'superchat.xml'), os.path.join(self.danmu_path, 'superchat.ass'),
        #         self.ffmpeg_logfile_hander)
        ret = concat(self.merge_conf_path, self.merged_file_path,
                     os.path.join(self.danmu_path, 'superchat.ass').replace("\\", "/"),
                     self.ffmpeg_logfile_hander)
        self.times.sort(key=lambda x: x[0])
        self.live_start = self.times[0][0]
        self.live_duration = (
                                     self.times[-1][0] - self.times[0][0]).total_seconds() + self.times[-1][1]
        return ret

    def __cut_video(self, outhint: List[str], start_time: int, delta: int) -> Union[
        subprocess.CompletedProcess, subprocess.CalledProcessError]:
        hours, remainder = divmod(start_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        output_file = os.path.join(
            self.outputs_dir,
            f"{self.room_id}_{self.global_start.strftime('%Y-%m-%d_%H-%M-%S')}_{hours:02}-{minutes:02}-{seconds:02}_{outhint}.mp4")
        cmd = f'ffmpeg -y -ss {start_time} -t {delta} -accurate_seek -i "{self.merged_file_path}" -c copy -avoid_negative_ts 1 "{output_file}"'
        try:
            ret = subprocess.run(cmd, shell=True, check=True,
                                 stdout=self.ffmpeg_logfile_hander, stderr=self.ffmpeg_logfile_hander)
            return ret
        except subprocess.CalledProcessError as err:
            traceback.print_exc()
            return err

    def cut(self, cut_points: List[Tuple[datetime.datetime, datetime.datetime, List[str]]],
            min_length: int = 60) -> bool:
        success = True
        duration = float(ffmpeg.probe(self.merged_file_path)
                         ['format']['duration'])
        for cut_start, cut_end, tags in cut_points:
            start = get_true_timestamp(self.times,
                                       cut_start) + self.config['spec']['clipper']['start_offset']
            end = min(get_true_timestamp(self.times,
                                         cut_end) + self.config['spec']['clipper']['end_offset'], duration)
            delta = end - start
            outhint = " ".join(tags)
            if delta >= min_length:
                ret = self.__cut_video(outhint, max(
                    0, int(start)), int(delta))
                success = success and isinstance(
                    ret, subprocess.CompletedProcess)
        return success

    def split(self, split_interval: int = 3600) -> bool:
        success = True
        if split_interval <= 0:
            shutil.copy2(self.merged_file_path, os.path.join(
                self.splits_dir, f"{self.room_id}_{self.global_start.strftime('%Y-%m-%d_%H-%M-%S')}_0000.mp4"))
            return success

        duration = float(ffmpeg.probe(self.merged_file_path)
                         ['format']['duration'])
        num_splits = int(duration) // split_interval + 1
        for i in range(num_splits):
            output_file = os.path.join(
                self.splits_dir, f"{self.room_id}_{self.global_start.strftime('%Y-%m-%d_%H-%M-%S')}_{i:04}.mp4")
            cmd = f'ffmpeg -y -ss {i * split_interval} -t {split_interval} -accurate_seek -i "{self.merged_file_path}" -c copy -avoid_negative_ts 1 "{output_file}"'
            try:
                _ = subprocess.run(cmd, shell=True, check=True,
                                   stdout=self.ffmpeg_logfile_hander, stderr=self.ffmpeg_logfile_hander)
            except subprocess.CalledProcessError:
                traceback.print_exc()
                success = False
        return success

    def run(self) -> bool:
        try:
            ret = self.pre_concat()
            success = isinstance(ret, subprocess.CompletedProcess)
            if success and not self.config.get('spec', {}).get('recorder', {}).get('keep_raw_record', False):
                if os.path.exists(self.merged_file_path):
                    utils.del_files_and_dir(self.record_dir)
            # duration = float(ffmpeg.probe(self.merged_file_path)[
            #                              'format']['duration'])
            # start_time = get_start_time(self.merged_file_path)
            # self.times.append((start_time, duration))
            # self.live_start = self.times[0][0]
            # self.live_duration = (
            #     self.times[-1][0]-self.times[0][0]).total_seconds()+self.times[-1][1]

            if not self.config.get('spec', {}).get('clipper', {}).get('enable_clipper', False) and not self.config.get(
                    'spec', {}).get('manual_clipper', {}).get('enabled', False):
                os.rmdir(self.outputs_dir)

            if not self.config.get('spec', {}).get('uploader', {}).get('record', {}).get('upload_record', False):
                os.rmdir(self.splits_dir)

            if self.config.get('spec', {}).get('clipper', {}).get('enable_clipper', False):
                danmu_list = parse_danmu(self.global_start, self.danmu_path)
                counted_danmu_dict = count(
                    danmu_list, self.live_start, self.live_duration,
                    self.config.get('spec', {}).get('parser', {}).get('interval', 60))
                cut_points = get_cut_points(counted_danmu_dict,
                                            self.config.get('spec', {}).get('parser', {}).get('up_ratio', 2.5),
                                            self.config.get('spec', {}).get('parser', {}).get('down_ratio', 0.75),
                                            self.config.get('spec', {}).get('parser', {}).get('topK', 5))
                ret = self.cut(cut_points, self.config.get('spec', {}).get(
                    'clipper', {}).get('min_length', 60))
                success = success and ret
            if self.config.get('spec', {}).get('manual_clipper', {}).get('enabled', False):
                danmu_list = parse_danmu(self.global_start, self.danmu_path)
                cut_points = get_manual_cut_points(danmu_list, self.config.get(
                    'spec', {}).get('manual_clipper', {}).get('uid', ""))
                ret = self.cut(cut_points, 0)
                success = success and ret
            if self.config.get('spec', {}).get('uploader', {}).get('record', {}).get('upload_record', False):
                ret = self.split(self.config.get('spec', {}).get('uploader', {})
                                 .get('record', {}).get('split_interval', 3600))
                success = success and ret
            return success
        except:
            traceback.print_exc()
            return False


if __name__ == "__main__":
    danmu_list = parse_danmu(time.time(), "data/data/danmu/22603245_2021-03-13_11-20-16")
    # counted_danmu_dict = count(
    #     danmu_list, datetime.datetime.strptime("2021-03-13_11-20-16", "%Y-%m-%d_%H-%M-%S"), (datetime.datetime.strptime("2021-03-13_13-45-16", "%Y-%m-%d_%H-%M-%S")-datetime.datetime.strptime("2021-03-13_11-20-16", "%Y-%m-%d_%H-%M-%S")).total_seconds(), 30)
    # cut_points = get_cut_points(counted_danmu_dict, 2.5,
    #                             0.75, 5)
    cut_points = get_manual_cut_points(danmu_list, "8559982")
    print(cut_points)
