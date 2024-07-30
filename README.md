# API-Calidad-Texto-LLM
La APi-Calidad-Texto-LLM fue ideada para integrarse al asistente holografica y educativa NadIA. Su funcionamiento radica en evaluar la calidad de las entradas y salidas del modelo GPT 3.5 Turbo accesado por el usuario atraves de NadIA.
La API mide las siguientes metricas:<br/>
<strong>Cohesion:</strong><br/>
$$Cohesion=\frac{NPPC}{NTPP}$$<br/>
<strong>Gramaticalidad:</strong><br/>
$$Gramaticalidad=1-\frac{NEG}{NTEGT}$$<br/>
<strong>Complejidad:</strong><br/>
$$Complejidad=\frac{NPU}{NTP}$$<br/>
<strong>Riqueza Lexica:</strong><br/>
$$TTR=\frac{V}{N}$$<br/>
<strong>Indice de Fernandez-Huerta:</strong><br/>
$$L=206.84-0.60P-1.02F$$<br/>
<strong>Indice Inflesz:</strong><br/>
$$I=206.835-\frac{62.3S}{P}-\frac{P}{F}$$<br/>
