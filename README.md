# Seq2Seq model

* Sequence to Sequence model은 sequence를 입력으로 받아 다른 도메인의 sequence를 출력으로 반환하는 모델으로 `Machine Translation, Chatbot,Text Summarization, Speech to Textr` 등에서 사용될 수 있습니다. 
* seq2seq 모델은 `RNN(Recurrent Neural Network),GRU(Gated Recurrent Units),LSTM(Long short Term Memory)` 로 구성할 수 있는데 Sequence의 길이가 길어질수록 적용하기 어렵다는 문제점이 생깁니다.
* 이에 등장한 개념이 _Attention_ 입니다

# Attention Mechanism

Attention의 기본 아이디어는 Decoder에서 출력 단어를 예측하는 매 time step마다, Encoder에서의 전체 입력 문장을 고려해 해당 time step에서 예측해야하는 단어와 연관이 있는 입력 단어 부분을 좀 더 _집중(Attention)_ 해서 보는 것입니다.



# Attention model

__Bidirectional GRU + Attention seq2seq structure__

<img src = https://images.velog.io/images/jyong0719/post/5d38f6a5-bd1f-4d71-8edd-8318168f38e3/image.png  width=70%;>

- $h^{<{t}'>} = (h^{\rightarrow{t}'},h^{\leftarrow{t}'})$
	- concat forward occurence and backward occurence

- $\alpha^{<t,{t}'>} = \text{ amount of 'attention' }y^{< t >} \text{ should pay to }h^{{t}'}$
- $\sum_{{t}'} \alpha^{<t,{t}'>} = 1$
	- attention weight, Alpha는 전체 sequence 중 각 timestep의 input을 고려하는 척도로 총 합은 1
- $C^{t} = \sum_{{t}'}\alpha^{<t,{t}'>}h^{<{t}' >}$

- $\alpha^{< t,{t}' >} = \frac{\exp(e^{< t,{t}' >})}{\sum_{{t}'=1}^{T_x}\exp(e^{< t,{t}' >})}$
  $e^{< t,{t}' >}$ is learn by values($s^{< t-1 >}, h^{< {t}' >}$) as input with small Neural network
  $T_X$ = input sequence length


# Transformers Motivation

처음 등장한 `Recurrent Neural Network`는 input sequence의 길이가 길어지면  vanishing gradient문제와 점점 더 학습하기 어렵다는 단점을 가지고 있었습니다, 이에 이를 해결하고자 `Long Short Term Memory` 모델과 이의 간소화 버전인 `Gated Recurrent Unit`의 모델들이 등장했습니다. 다만 이후의 등장한 모델들 역시 Sequence length가 커질수록 연산의 복잡도가 증가하는 문제점을 가지고 있습니다.

Rnn으로 이루어진 seq2seq 구조에서는 Encoder는 input sequence를 하나의 벡터표현으로 압축하고, 디코더는 이 벡터 표현을 통해서 output sequence를 생성합니다. 이러한 구조는 encoder가 input sequence를 하나의 벡터로 압축하는 과정에서 정보가 일부 손실된다는 단점이 있었고, 이를 보정하기 위해 attention이 사용되었습니다. 

`Transformer`는 attention을 RNN의 보정을 위한 용도가 아니라 Encoder,Decoder를 아예 attention으로 구성해보는 방법에서 시작되었습니다.


# Self Attention

$A(q,K,V)$ = attention based vector representation of a word 

RNN Attention : $\alpha^{< t,{t}' >} = \frac{\exp(e^{< t,{t}' >})}{\sum_{{t}'=1}^{T_x}\exp(e^{< t,{t}' >})}$

Transformers Attention : $A(q,K,V) = \sum_i\frac{\exp(e^{< q\dot{k^{< i >} >}} )}{\sum_j\exp(e^{< q\dot{k^{< j >} >}} )}v^{ < i >}$ ; `especially scaled dot product Attention`

__$A^3$ 을 예측한다고 가정하는 경우__

<img src = https://images.velog.io/images/jyong0719/post/dde9baa1-a90c-4c4d-8649-69a4da0c1243/image.png width=60%>

input vector : $x^3$ = word embedding vector
$q,k,v$ is a leared matrix

$q^t = W^q \cdot x^t$
$k^t = W^k \cdot x^t$
$v^t = W^v \cdot x^t$

$W^q,W^k,W^v$ are parameters of this learning algorithm

q^3 may represent a question 

Ex) 
Suppose that French2English machine translation model

Y = Jane visits Africa in September
A(attention score value) = A1, A2, A3, A4, A5
X = Jane visite I'Afrique en septembre 


When computing $A^3$(Africa), q3 may represent 'what's happening there?'

inner product between $q^3,k^1$ -> tell how good as an answer

if $k$ represents 
- $k^1$ : person
- $k^2$ : action
- and $k^3,k^4$.. so on

may find that $q^3,k^2$ inner product has the largest value that might suggest that 'visit' is the most relevant context for $q^3$('what's happening in Africa?')

    
$v$ : value represent to plug in how visit should be represented within $A^3$ within the representation of Africa

$A(q,K,V) = \sum_i\frac{\exp(e^{< q\dot{k^{< i >} >}} )}{\sum_j\exp(e^{< q\dot{k^{< j >} >}} )}v^{ < i >} = softmax(\frac{QK^T}{\sqrt{d_k}})v$

## Kinds of self-attention 

|Name|score function($score(q_t,k_{ { t }'})$) | Defined by|
|---|---|---|
|dot|$q_t^T k_{{t}'}$|Luong et al. (2015)|
|scaled dot|$\frac{QK^T}{\sqrt{d_k}}$|Vaswani et al. (2017)|
|general|$q_t^TW_ak_{{t}'}$ // $W_a$는 학습 가능한 weight matrix |Luong et al. (2015)|
|concat| $W_a^T tanh(W_bq_t + W_ck_{{t}'})$      |Bahdanau et al. (2015)|
|location-base|$a_t = softmax(W_aq_t)$ // $a_t$ 산출시에 $q_t$만 사용하는 방법. |Luong et al. (2015)|

score function : input of softmax



# Multi-Head Attention
<img src = https://images.velog.io/images/jyong0719/post/3e300a29-ed54-4f0f-95d6-902f070b7d35/image.png  width=70%;>


하나의 self-attention {$W_1^QQ,W_1^KK,W_1^VV$}을 head_1라고 정의하면 $head_i = Attention(W_i^QQ,W_i^KK,W_i^VV)$

$head_1 : W_1^Q,W_1^K,W_1^V$ as being learned to help ask and answer the question `what happening there?`
$head_2 : W_2^Q,W_2^K,W_2^V$ maybe other ask `when something happening?`
$head_3$ : `who?` and $heads_i...$ so on


$MultiHead(Q,K,V) = concat(head_1head_2head_3 ... head_h)$; $h=$ number of heads


self-attention을 여러번 수행하는 연산으로 구현 자체는 For loop을 사용할 수 있고, model의 capacity를 늘려주는 효과인 것 같다. 각각의 head는 모두 독립적이기 때문에 병렬적인 연산으로 한번에 연산이 가능하다.





*참고
- coursera(Sequence models)
- wikidocks(딥 러닝을 이용한 자연어 처리 입문)
