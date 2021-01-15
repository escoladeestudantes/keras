<div style="text-align:center"><a href="https://www.youtube.com/watch?v=lJRXDKFBiTc"><img src="https://i.imgur.com/qZtrUeb.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Reconhecimento de expressão facial com CNN de Sefik Serengil treinada no dataset Fec2013</h3>

<p>Reconhecimento de expressão facial usando CNN a partir da implementação compartilhada por Sefik Ilkin Serengil com Keras 2.3.0 e TensorFlow 2.0 como backend.</p>

<p>Baixar os arquivos: </p>
<ul>
<li>https://github.com/serengil/tensorflow-101/blob/master/model/facial_expression_model_structure.json</li>
<li>https://github.com/serengil/tensorflow-101/blob/master/model/facial_expression_model_weights.h5</li>
<li>https://github.com/serengil/tensorflow-101/blob/master/python/facial-expression-recognition-from-stream.py</li>
<li>https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml</li>
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
  <li>https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/</li>
  <li>https://sefiks.com/2018/01/10/real-time-facial-expression-recognition-on-streaming-data/</li>
  <li>https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error</li>
  <li>https://github.com/escoladeestudantes/opencv/tree/main/04_Baixar_classificadores_OpenCV</li>
</ul>
