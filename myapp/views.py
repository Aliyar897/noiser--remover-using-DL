from django.shortcuts import render
from django.http import HttpResponse
from .args import parser
from .prepare_data import create_data
from .train_model import training
from .ml_model import MLModel
from .prediction_denoise import prediction

# Create your views here.
def home(request):
    
    return render(request, 'home.html',{})

def upload(request):
    mymodel = MLModel()
    file_name = ''
    # result = mymodel.prediction(?)
    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_input_prediction = request.FILES['audio_file']
        mymodel = MLModel()
        mymodel.prediction(audio_input_prediction)
        # data = {
        #     'filename':audio_input_prediction,
        # }
        # data = audio_input_prediction.name
        filename = audio_input_prediction
        deniosedfile = 'denoised'
        return render(request, 'home.html', {'filename': filename, 'denoised':deniosedfile})
    else:
        return  HttpResponse('file not found!')
    # return HttpResponse('this is predict view')


    # return HttpResponse('this is the result: ', result)

    
    
    # weights_path = 'C:\\Users\\user\\Desktop\\internship\\speech enhancement repo\\Speech-enhancement\\weights\\model_unet.json'
    # weights_path = weights_path.encode('utf-8')
    
    # name_model = 'C:\\Users\\user\\Desktop\\internship\\speech enhancement repo\\Speech-enhancement\\weights\\model_best.h5'
    # name_model = name_model.encode('utf-8')
    # # audio_dir_prediction ='C:\\Users\\user\\Desktop\\internship\\speech enhancement repo\\noise remover\\noiseremover\\myapp\\demo_data\\test'
    # # audio_dir_prediction = audio_dir_prediction.encode('utf-8')
    # dir_save_prediction = 'C:\\Users\\user\\Desktop\\internship\\speech enhancement repo\\noise remover\\noiseremover\\myapp\\demo_data\\save_predictions'
    # dir_save_prediction = dir_save_prediction.encode('utf-8')
    # # audio_input_prediction = 'C:\\Users\\user\\Desktop\\internship\\speech enhancement repo\\noise remover\\noiseremover\\myapp\\demo_data\\test\\noisy_voice_long_t3.wav'
    # audio_output_prediction = 'C:\\Users\\user\\Desktop\\internship\\speech enhancement repo\\noise remover\\noiseremover\\myapp\\demo_data\\save_predictions\\denoise_t3.wav'
    # # audio_output_prediction = audio_output_prediction.encode('utf-8')
    # sample_rate = 8000
    # min_duration = 1.0
    # frame_length =  8064
    # hop_length_frame = 8064
    # n_fft = 255
    # hop_length_fft =63
    # if request.method == 'POST' and request.FILES.get('audio_file'):
    #     audio_input_prediction = request.FILES['audio_file']
    #     mymodel = MLModel()
    #     mymodel.prediction(audio_input_prediction)
    #     return render(request, 'home.html')
    # else:
    #     return  HttpResponse('file not found!')
    # return HttpResponse('this is predict view')

# def upload(request):
#     print('This is the response :', request)
# #     args = parser.parse_args()    
# #             #Example: python main.py --mode="prediction"
# #             #path to find pre-trained weights / save models
# #     weights_path = args.weights_folder
# #             #pre trained model
# #     name_model = args.name_model
# #             #directory where read noisy sound to denoise
# #     audio_dir_prediction = args.audio_dir_prediction
# #             #directory to save the denoise sound
# #     dir_save_prediction = args.dir_save_prediction
# #             #Name noisy sound file to denoise
# #     audio_input_prediction = args.audio_input_prediction
# #             #Name of denoised sound file to save
# #     audio_output_prediction = args.audio_output_prediction
# #             # Sample rate to read audio
# #     sample_rate = args.sample_rate
# #             # Minimum duration of audio files to consider
# #     min_duration = args.min_duration
# #             #Frame length for training data
# #     frame_length = args.frame_length
# #             # hop length for sound files
# #     hop_length_frame = args.hop_length_frame
# #             #nb of points for fft(for spectrogram computation)
# #     n_fft = args.n_fft
# #             #hop length for fft
# #     hop_length_fft = args.hop_length_fft
# #     prediction = prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
# #         audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft)
# #     data ={
# #         prediction:'prediction'
# #     }

#     return render(request, 'home.html')