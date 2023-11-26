import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont


def main():
    loaded_model = keras.models.load_model("ModelOutput/modeled.keras")
    predict_image(loaded_model, "Data/Test/Medium/Crack__20180419_06_19_09,915.bmp")
    predict_image(loaded_model, "Data/Test/Large/Crack__20180419_13_29_14,846.bmp")


def predict_image(model, image_path):
    labels = ["Large", "Medium", "Small", "None"]
    input_image = Image.open(image_path)
    input_image = input_image.convert("RGB")
    test_img = keras.utils.load_img(image_path, target_size=(100, 100))
    img_array = keras.utils.img_to_array(test_img)
    img_array = tf.expand_dims(img_array, 0) 
    img_array = img_array/255.
    predictions = model.predict(img_array)
    for i,prob in enumerate(predictions[0]):
        print(labels[i], ": ", prob)

    # set_font = ImageFont.load_default()
    # set_font.size = 100

    set_font = ImageFont.truetype("UbuntuMono-R.ttf", size=100)
    

    draw_image = ImageDraw.Draw(input_image)
    draw_image.text((1400,1700), "Large: " + str(round(predictions[0][0] * 100)) + "%", font = set_font, fill = 'green')
    draw_image.text((1400,1775), "Medium: " + str(round(predictions[0][1] * 100)) + "%", font = set_font, fill = 'green')
    draw_image.text((1400,1850), "Small: " + str(round(predictions[0][2] * 100)) + "%", font = set_font, fill = 'green')
    draw_image.text((1400,1925), "None: " + str(round(predictions[0][3] * 100)) + "%", font = set_font, fill = 'green')
    filename = image_path.split('/')[-1]

    input_image.save("ModelOutput/Predictions/Pred_" + filename, "jpeg", quality = 75)



if __name__ == "__main__":
    main()