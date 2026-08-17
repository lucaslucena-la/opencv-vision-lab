# Vision Lab

Laboratório de visão computacional com controle por gestos. A webcam captura a
imagem, os modelos de inteligência artificial processam em tempo real e você
interage com o **dedo indicador** (sem mouse nem teclado, exceto pela tecla
`ESC` para sair).

O projeto funciona em **Python 3.11** e usa OpenCV, MediaPipe, TensorFlow e
Pillow (dependências listadas em `requirements.txt`).

---

## 1. O programa `menu.py` e seus sub-componentes

`menu.py` é a **porta de entrada** do sistema. Ele abre a câmera, mostra uma
tela com 4 cartões e deixa você escolher qual experiência abrir apontando o
dedo.

### Como funciona o menu

- A câmera liga e aparece a tela **"VISION LAB"** com 4 cartões (Frutas, Mãos,
  Pinça, Jogo).
- Um **cursor** acompanha a ponta do seu dedo indicador (detectado pelo
  MediaPipe `HandLandmarker`).
- Ao manter o dedo sobre um cartão por **2 segundos**, um arco de progresso
  aparece e, ao completar, o programa correspondente abre.
- Dentro de cada experiência, há um botão **"X"** no canto superior direito:
  aponte o dedo nele por 1,5 s para voltar ao menu.
- A tecla `ESC` encerra a execução atual.

Os sub-componentes importados pelo menu são:

| Cartão | Arquivo | O que faz |
|--------|---------|-----------|
| 🍎 Frutas | `fruits.py` | Aponta a câmera para uma fruta e o modelo `EfficientNetB0` (treinado no ImageNet) diz qual é. Mostra o nome com confiança e o emoji correspondente sobre a imagem. |
| 🖐️ Mãos | `hands.py` | Reconhece gestos da mão (`Thumb_Up`, `Victory`, `Closed_Fist`, etc.) e desenha o emoji equivalente. Também detecta 3 gestos personalizados (🤘, 🖕, 🤙) calculando dedos estendidos/dobrados. |
| 🍪 Pinça | `pinch.py` | Um biscoito 🍪 na tela. Com uma mão, junta polegar e indicador (pinça) para **arrastá-lo**. Com as duas mãos em pinça, afaste ou aproxime para **aumentar ou diminuir** o biscoito (zoom). |
| 🎮 Jogo | `handgame.py` | Mini-jogo "pegue as moedas". Você escolhe suas iniciais por gesto, clica em "JOGAR" e coleta moedas por 10 segundos movendo o dedo. O placar é salvo em `ranking.txt`. |

> Observação: `menu.py` carrega os modelos pesados uma única vez e exibe uma
> tela "Carregando modelos..." enquanto isso acontece.

---

## 2. O arquivo de emoções (`emotions.py`)

`emotions.py` é um script **independente** que reconhece a emoção de rostos em
tempo real.

### Como funciona

1. **Detecção de rosto** — usa o classificador Haar Cascade do OpenCV
   (`haarcascade_frontalface_default.xml`) para encontrar rostos no frame.
2. **Pré-processamento** — recorta o rosto, redimensiona para 64×64 pixels,
   converte para escala de cinza e normaliza os valores (0 a 1).
3. **Classificação** — o modelo de rede neural
   `fer2013_mini_XCEPTION.102-0.66.hdf5` (treinado no dataset FER2013) prevê a
   emoção entre 7 categorias.
4. **Exibição** — desenha um retângulo ao redor do rosto, escreve a emoção
   detectada e mostra a **probabilidade** (em %) de cada uma das 7 emoções em
   uma lista no canto da tela.

As 7 emoções reconhecidas (rótulos FER2013):

1. Raiva
2. Nojo
3. Medo
4. Feliz
5. Triste
6. Surpreso
7. Neutro

### Versão avançada (`full.py`)

Existe também o `full.py`, uma versão mais completa que combina **emoção +
gênero + idade** em uma interface estilo "sci-fi" (HUD neon, barras de
probabilidade, scanlines). Ele usa três modelos juntos:

- Emoção → `fer2013_mini_XCEPTION.102-0.66.hdf5`
- Gênero → `gender_net.caffemodel` (rede Caffe)
- Idade → `age_net.caffemodel` (rede Caffe)

No `full.py`, as teclas trocam o modo de exibição:
`[0]` tudo, `[1]` só emoção, `[2]` só gênero, `[3]` só idade, `[ESC]` sair.

---

## 3. Tutorial — como abrir (tudo já instalado)

Pré-requisito: Python 3.11, as bibliotecas de `requirements.txt` e os arquivos
de modelo (`.task`, `.hdf5`, `.caffemodel`, `.prototxt`) já presentes na pasta.
Nada precisa ser instalado.

### Passo a passo

1. **Abra o terminal** (no Windows, use o PowerShell ou o Prompt; no Linux/macOS,
   o terminal padrão).

2. **Entre na pasta do projeto**:

   ```bash
   cd /home/luigr/opencv-vision-lab
   ```

   *(ajuste o caminho para onde o projeto está no seu computador)*

3. **(Opcional) Ative o ambiente virtual**, se o projeto tiver um:

   ```bash
   # Linux/macOS
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

4. **Abra o menu**:

   ```bash
   python menu.py
   ```

   Aparece a tela "VISION LAB". Aponte o dedo indicador e segure 2 segundos no
   cartão desejado.

### Abrir um programa específico

Para rodar qualquer script sem passar pelo menu, use `python` + nome do arquivo:

```bash
python emotions.py   # reconhece emoções
python full.py       # emoção + gênero + idade (HUD sci-fi)
python fruits.py     # classifica frutas
python hands.py      # gestos → emojis
python pinch.py      # pinça para arrastar/zoom no biscoito
python handgame.py   # mini-jogo das moedas
```

### Controles básicos

- **Dedo indicador** → move o cursor (e mantém por 2 s para selecionar).
- **Botão "X"** (canto superior direito) → apontar 1,5 s para voltar.
- **`ESC`** → encerra qualquer programa na hora.

### Se a câmera não abrir

O código testa os índices de câmera `0, 1, 2` automaticamente. Se nenhum
funcionar, verifique se a webcam está conectada e se nenhum outro aplicativo a
está usando (chamadas de vídeo, por exemplo).
