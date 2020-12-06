import SpeakerIdentification as si

if __name__=="__main__":
    while True:
        choice = int(input("\n 1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n Chon: "))
        if (choice == 1):
            si.record_audio_train()
        elif (choice == 2):
            si.train_model()
        elif (choice == 3):
            si.record_audio_test()
        elif (choice == 4):
            si.test_model()
        if (choice > 4):
            exit()