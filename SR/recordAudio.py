import os
import pyaudio
import CONFIG as c
import wave
import time


def record_audio(Quantity_file, record_seconds, file_text, dir, test=False):
    """
    :param Quantity_file: số lượng file ghi âm
    :param record_seconds: thời gian ghi âm
    :param file_text: file name lưu vào text
    :param dir: path save file
    :param test: default=False
    :return: void
    """
    Name = (input("Nhập tên của bạn: "))

    for count in range(Quantity_file):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=c.FORMAT, channels=c.CHANNELS, rate=c.RATE, input=True,
                            input_device_index=c.DEVICE_INDEX, frames_per_buffer=c.CHUNK)

        print(f"Bắt đầu ghi file: ({count + 1}/{Quantity_file})")
        Record_frames = []
        for i in range(0, int(c.RATE / c.CHUNK * record_seconds)):
            data = stream.read(c.CHUNK)
            Record_frames.append(data)

        print(f"Xong {count + 1}/{Quantity_file}.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        if test == True:
            out_filename = Name+"-test" + str(count) + '.wav'
        else:
            out_filename = Name + "-sample" + str(count) + '.wav'

        wav_output_filename = os.path.join(dir, out_filename)
        if count == 0:
            train_file_list = open(file_text, 'w')
            train_file_list.write(out_filename + '\n')
            wavefile = wave.open(wav_output_filename, 'wb')
            wavefile.setnchannels(c.CHANNELS)
            wavefile.setsampwidth(audio.get_sample_size(c.FORMAT))
            wavefile.setframerate(c.RATE)
            wavefile.writeframes(b''.join(Record_frames))
            wavefile.close()
        else:
            train_file_list = open(file_text, 'a')
            train_file_list.write(out_filename + '\n')
            wavefile = wave.open(wav_output_filename, 'wb')
            wavefile.setnchannels(c.CHANNELS)
            wavefile.setsampwidth(audio.get_sample_size(c.FORMAT))
            wavefile.setframerate(c.RATE)
            wavefile.writeframes(b''.join(Record_frames))
            wavefile.close()
        time.sleep(5)


def record_one_file(file_name):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=c.FORMAT, channels=c.CHANNELS,
                        rate=c.RATE, input=True, input_device_index=c.DEVICE_INDEX,
                        frames_per_buffer=c.CHUNK)
    print("Bắt đầu ghi âm: ")
    Recordframes = []
    for i in range(0, int(c.RATE / c.CHUNK * c.RECORD_SECONDS)):
        data = stream.read(c.CHUNK)
        Recordframes.append(data)
    print("Ghi âm xong! Đang lưu file")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    WAVE_OUTPUT_FILENAME = os.path.join(c.TEST_SET, file_name)
    trainedfilelist = open(c.FILE_NAME_TEST, 'a')
    trainedfilelist.write(file_name + "\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(c.CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(c.FORMAT))
    waveFile.setframerate(c.RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()
    print("Lưu xong! Bắt đầu test")
    return WAVE_OUTPUT_FILENAME
