<div style="text-align:center"><a href="https://www.youtube.com/watch?v=tFQv_zMiWxY"><img src="https://i.imgur.com/3e1bq4T.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Reconhecimento facial com VGG-Face usando TensorFlow 2.0 como backend</h3>

<p>Reconhecimento facial usando VGG-Face a partir da implementação compartilhada por Sefik Ilkin Serengil com Keras 2.3.0 e TensorFlow 2.0 como backend.
</p>

<p>Baixar o arquivo<p/>
<ul>
  <li>https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing</li>
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

<b>Referência<b/>
<ul>
  <li>https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/</li>
  <li>https://github.com/serengil/tensorflow-101/blob/master/python/deep-face-real-time.py</li>
  <li>http://www.robots.ox.ac.uk/~vgg/software/vgg_face/</li>
</ul>
