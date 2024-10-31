# Model with **waveforms**

Date : June 01 <br>
Writer : 박성민

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_section import train_test_split
import matplotlib.pyplot as plt

# 파형 모습으로 학습하는 CNN모델 일단 케라스랑 텐서플로우로 만들어봄
# 데이터 가공 추가해야함
# 데이터 훈련셋 검증셋으로 나누기 그 비율은 일단 임의로 0.2로 햇긴햇는데 이건 상의해봐야댈듯
(x_train_all,y_train_all) =
x_train,x_val,y_train,y_val = train_test_split(x_train_all,y_train_all,stratify=y_train_all,test_size=0.2,random_state=42)
y_train_encoded = tf.keras.utils.to_categorical(y_train)  #원핫인코딩 
y_val_encoded = tf.keras.utils.to_categorical(y_val)    
x_train = x_train.reshape(-1,1600,600,1)  #사진크기 바뀌면 바꾸면댐
x_val = x_val.reshape(-1,1600,600,1)
x_train = x_train/255
x_val = x_val/255 
# /255 이거 표준화 헀긴했는데 이거 좀 애매함 더좋은 방법있으면 바꾸면댈듯

# 신경망 여기 추가해야될거 머잇지 일단 새임패딩썻고 풀링층도면 2바2로 햇음
# evoke20으로 했는데 이거 의존도 높아지면 낮추면 대고
conv1=tf.keras.Sequential()
conv1.add(Con2D(10,(3,3),activation='relu',padding='same',input_shape=(1600,600,1)))
conv1.add(MaxPooling2D(2,2))
conv1.add(Flatten())
conv1.add(Dense(10,activation='relu'))  # 여기 뉴런갯수 몇개로 해야할지 모르겟음 
conv1.add(Dense(1,activation='softmax'))
conv1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = conv1.fit(x_train,y_train_encoded,epochs=20,validation_data=(x,val,y_val_encoded))  #학습

# 손실함수 분석용 과소적합은 아마 안날듯 데이터 개많아서?
# 과대적합나면 드랍아웃이나 L2 규제 추가해야댐 일단 손실함수 개형보고 판단 ㄱ
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy','val_accuracy'])
plt.show()

```

이거 근데 생각해보니까 그래프 x축 y축 주석들 다 짜르고 그래프모양만 입력데이터로 넣어줘야댐 <br>

어짜피 그래프 개형으로만 학습할거라 계산량만 늘어나고 의미없는 입력값임
그래서 사진크기 1600*600 아니고 좀 줄어들듯 나중에 모델수정 ㄱ <br>

보니까 실제음성도 파형이 덜? 특이한것들 있는데 그런 데이터는 정확도 좀 떨어지더라도 
감안해야할듯 다른 더 좋은 방법생각나면 바로 갈아타도댐 <br>

추가적으로 해봐야 할거는 음파 특징 진동수 진폭 파장 등등 특성몇개 더 추가해서 추세선 그려보는거
로 다른방법 어떻게 할지 정보 얻어볼 순 있을듯 근데 그거 어케하는지 몰겟음.. <br>

근데 결정적인 문제가 이거 방음 안댄 데이터 들어오면 실제음성이더라도 얘네가 외부소음땜에 
파형 잘못판단할듯....... 이거 어케하냐 <br>


