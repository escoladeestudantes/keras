<div style="text-align:center"><a href="https://www.youtube.com/watch?v=pRkLu39dFtg"><img src="https://i.imgur.com/QAO1bk1.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Estimativa da pose da cabeça com a implementação de Arnaldo Gualberto</h3>

<p>Estimativa da pose da cabeça (head pose estimation) com a implementação de Arnaldo Gualberto (usuário arnaldog12 no GitHub) usando o detector de pontos faciais da biblioteca Dlib e DNN treinada com Keras 2.3.0 e TensorFlow 2.0 como backend.</p>

<p>Clonar o repositório do detector de Arnaldo Gualberto (link nas Referências).<p/>


<p>Baixar os arquivos <b>shape_predictor_68_face_landmarks.dat.bz2</b> e <b>mmod_human_face_detector.dat.bz2</b> do link:</p>

<ul>
<li>www.dlib.net/files</li>
</ul>

Corrigir erro:
<p><b>...</p></b>
<p><b>Function call stack:</p></b>
<p><b>keras_scratch_graph</p></b>

<p>Adicionar no código</p>

```
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

<b>Referências</b>
<ul>
  <li>https://github.com/kairess/eye_blink_detector</li>
  <li>https://github.com/Guarouba/face_rec</li>
  <li>https://github.com/arnaldog12/Deep-Learning/tree/master/problems/Regressor-Face%20Pose</li>
  <li>https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error</li>
</ul>
