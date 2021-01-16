<div style="text-align:center"><a href="https://www.youtube.com/watch?v=sG3xrAIJhU0"><img src="https://i.imgur.com/buko7f7.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Predição de idade e gênero com o modelo VGG-Face – Sefik Serengil</h3>

<p>Predição de idade e gênero com o modelo VGG-Face a partir da implementação compartilhada por Sefik Ilkin Serengil com Keras 2.3.0 e TensorFlow 2.0 como backend.</p>

<p>Baixar os arquivos: </p>
<ul>
<li>https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view</li>
<li>https://drive.google.com/file/d/1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk/view</li>
<li>https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml</li>
</ul>


<p>Corrigir erro:</p>
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
  <li>https://sefiks.com/2019/02/13/apparent-age-and-gender-prediction-in-keras/</li>
  <li>https://github.com/serengil/tensorflow-101/blob/master/python/age-gender-prediction-real-time.py</li>
</ul>
