# doing_dacon
call -> 데이콘 Basic 전화 해지 여부 분류 AI 경진대회
https://drive.google.com/drive/folders/1r8TYvCPXc3Usn1ufLexpYcf2rGKYVRg1?usp=share_link -> Stacking input file

python run_automl.py -> automl 결과 생성(with oof prediction)

cd ./call_tanular

python main.py --stacking_file current_best_stacking_input_ --hidden 512 --n_layer 5 --learning_rate 1e-3 -> current best prediction
