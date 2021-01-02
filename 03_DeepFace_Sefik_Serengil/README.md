<div style="text-align:center"><a href="https://www.youtube.com/watch?v=ymNdThkoWcY"><img src="https://i.imgur.com/jGN6Sey.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Reconhecimento facial com DeepFace usando TensorFlow 2.0 como backend</h3>

<p>Reconhecimento facial usando DeepFace a partir da implementação compartilhada por Sefik Ilkin Serengil com Keras 2.3.0 e TensorFlow 2.0 como backend.</p>

<p>Baixar e extrair o arquivo<p/>
<ul>
  <li>https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip</li>
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
  <li>https://sefiks.com/2020/02/17/face-recognition-with-facebook-deepface-in-keras/</li>
  <li>https://github.com/swghosh/DeepFace/releases</li>
</ul>
