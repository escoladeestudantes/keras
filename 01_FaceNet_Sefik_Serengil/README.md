<div style="text-align:center"><a href="https://www.youtube.com/watch?v=6uCJ-FUy0rA"><img src="https://i.imgur.com/anFkfUg.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Reconhecimento facial com FaceNet usando TensorFlow 2.0 como backend</h3>

<p>Baixar arquivos<p/>
<ul>
  <li>https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view</li>
  <li>https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json</li>
  <li>https://github.com/serengil/tensorflow-101/blob/master/model/inception_resnet_v1.py</li>
</ul>

<p>Corrigir erro:</p>
<p><b>...</p></b>
<p><b>SystemError: unknown opcode</p></b>

<p>Comentar (ou excluir) a linha:</p>

```
model = model_from_json(open("facenet_model.json", "r").read())
```


<p>Adicionar no código</p>

```
from inception_resnet_v1 import *
model = InceptionResNetV1()
```

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
  <li>https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/</li>
</ul>
