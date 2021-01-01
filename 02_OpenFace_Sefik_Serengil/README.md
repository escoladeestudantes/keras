<div style="text-align:center"><a href="https://www.youtube.com/watch?v=bQvO11Xg_l8"><img src="https://i.imgur.com/ctTkxVO.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Reconhecimento facial com OpenFace usando TensorFlow 2.0 como backend</h3>

<p>Baixar arquivo<p/>
<ul>
  <li>https://drive.google.com/file/d/1LSe1YCV1x-BfNnfb7DFZTNpv_Q9jITxn/view</li>
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
  <li>https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/</li>
</ul>
