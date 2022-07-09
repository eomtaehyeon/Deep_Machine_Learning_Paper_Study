# Attention is all you need Review

[[https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf) ](https://arxiv.org/pdf/1706.03762.pdf)

[https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf) 

# Description

- Transformer, Attention 만으로 시퀀셜 데이터를 분석하여 병렬화와 연산 속도 향상을 가능하게 한 새로운 모델 제시.

# 핵심 요약

- Seq2Seq 과 Attention 을 결합한 모델에서 한층 더 발전한 모델입니다.
- Recurrent model(재귀 구조)없이 Self-attention 만으로 구성한 첫번째 모델입니다.
- 재귀 구조 제거로 모델을 병렬화(Parallelization)하여 자연 언어 처리 학습/추론 시간을 획기적으로 단축시켰습니다.

# 딥러닝 기반의 기계번역 발전 과정

![](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled.png)

- 1986년에 RNN(Recurrent Neural Networks) 모델이 제안.
- 10년 후 LSTM(Long Short-term memory)모델이 등장하였습니다. LSTM을 활용하여 다양한 시퀀스 정보를 모델링 하여 주가 예측, 주기함수 예측 등이 가능했다.
- 2014년에는 LSTM을 활용한 딥러닝기반 기술로 Seq2Seq 등장 하였습니다. Seq2Seq는 고정된 크기의 context vector를 사용하여 번역을 수행하는 것을 제안하였습니다.
- 2015년에는 Seq2Seq 성능적 한계를 Attention 기법을 적용하여 성능을 더욱 끌어 올렸다.
- 2017년에는 Transformer논문에서 RNN 자체를 사용하지 않고 Attention 만 사용하고, Attention에 의존하는 아키텍쳐를 설계
    
    ⇒ 성능이 훨씬 좋아지는 것을 확인하였다.
    
- 최근 추세는 RNN기반의 아키텍쳐 자체를 사용하지 않고 Attention에 집중하는 방향으로 흘러감.
    
    ⇒ RNN을 활용한 논문들도 많이 존재한다. 
    
- 입력 시퀀스 전체에서 정보를 추출하는 방향으로 발전하고 있습니다.

# 기존 Seq2Seq 모델

![](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%201.png)

- 한쪽의 시퀀스에서 다른 한쪽의 시퀀스를 만든다는 의미에서 Seq2Seq모델이라고 부를 수 있다.
- context vector v에 소스 문장의 정보를 압축한다.
- 병목현상이 발생하여 성능 하락의 원인이 된다.
- 인코더에서 히든스테이트값은 소스 문장의 전체를 대표하는 context vector로서 문맥적인 정보를 담고 있다고 가정한다.
- 디코더에서 매번 히든스테이트값을 만들고 갱신하여 end of seq가 나올때까지 반복한다.

![](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%202.png)

- 문제인식 및 해결방안

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%203.png)

- 히든스테이트를 출력값으로서 별도의 배열에다가 저장하고 참고한다.
- 히든스테이트를 통해 어떤 단어에 초점을 줄지 가중치값을 만들어 반영하려고 한다.
- 매번 출력할때 마다 소스문장에서 나왔던 모든 출력값을 참고하는 방법으로 성능을 향상시킴.

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%204.png)

- 시각화를 통해 단어의 가중치를 확인 할 수 있다.

# 트랜스포머

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%205.png)

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%206.png)

### 인코더(Encoder)

- $N$(=6)개의 동일한 레이어로 구성
- 각 레이어는 2개의 하위 레이어로 구성
    - Multi-head self-attention
    - position-wise fully connected feed-forward
- 하위 레이어를 거칠 때마다 Residual connection(Resnet) 과 layer normalization 을 실행
- 각 레이어 출력의 크기는 $d_{model}$(=512)로 고정

### 디코더(Decoder)

- 인코더와 같이 $N$(=6)개의 동일한 레이어로 구성
- 인코더와 동일한 2개의 하위 레이어에 한가지를 더 추가하여 3개의 하위 레이어로 구성
    - Multi-head self-attention
    - position-wise fully connected feed-forward
    - 인코더의 출력으로 실행하는 multi-head attention
- 순차적으로 결과를 만들어 낼 수 있도록 Self-attention 레이어에 Masking 을추가 : $i$ 번째 출력을 만들 때, $i$번째보다 앞선 출력($i-1, i-2,\dots$) 만을 참고하도록 함

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%207.png)

- 트랜스포머는 임베딩 디멘션 값을 512로 고정한다.

| I | 여기는  | 임베딩 | 디멘션 |  | 여기까지 |
| --- | --- | --- | --- | --- | --- |
| am |  |  |  |  |  |
| a |  |  |  |  |  |
| teacher |  |  |  |  |  |
- 위치 정보를 포함한 임베딩으로 포지셔널 인코딩을 사용

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%208.png)

- 이후 트랜스포머에서는 셀프 어텐션 진행한다.
- 단어 서로에게 어텐션 스코어를 구해서 연관성의 정보를 학습하도록 한다.

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%209.png)

- 잔여 학습은 어텐션을 거치지 않은 정보를 가져와서 학습하도록 한다.
- 잔여 학습은 이미지 분류네트워크에서 사용되는 기법으로 전체 네트워크는 학습난이도는 낮고 초기 모델수련속도가 높게 된다.

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%2010.png)

- 이 과정을 반복합니다.

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%2011.png)

- 입력 값과 출력 값의 디멘져는 동일하다.

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%2012.png)

![Untitled](/Transformer/Attention%20is%20all%20you%20need%20Review/Untitled%2013.png)


# Reference :
- 나동빈님의 Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)  
 : https://www.youtube.com/watch?v=AA621UofTUA
- 나동빈님의 코드리뷰  
 : https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb
- 허민석님의 트랜스포머 (어텐션 이즈 올 유 니드)  
 : https://www.youtube.com/watch?v=mxGCEWOxfe8&t=369s
