<div style="text-align:center"><a href="https://www.youtube.com/watch?v=7k1uYiJ5G3Q"><img src="https://i.imgur.com/A4V2bQH.jpg" title="source: imgur.com" /></a></div>

<h3>Keras – Predição do estado do olho com as CNNs de Lee e Eetveldt com TensorFlow 2.0 como backend</h3>

<p>Predição do estado do olho para detecção de piscada (eye blink detection) com as redes neurais convolucionais (CNNs) de Taehee Lee e Jordan Van Eetveldt usando Keras 2.3.0 e TensorFlow 2.0 como backend.
</p>

<p>Clonar os repositórios dos detectores de Taehee Lee e Jordan Van Eetveldt (link nas Referências).<p/>

<p> <b>Taehee Lee</b> necessita da biblioteca scipy que tenha imread, imresize e imsave. O vídeo foi feito com a versão 1.1.0 conforme está destacado no arquivo <b>eye_status.py</b>. </p>


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
  <li>https://stackoverflow.com/questions/9298665/cannot-import-scipy-misc-imread</li>
  <li>https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error</li>
</ul>
