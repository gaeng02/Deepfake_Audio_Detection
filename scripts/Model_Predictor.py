'''
This script predicts the label of real test data.

Step 1. Load model.
Step 2. Predict.
Step 3. Write.
'''

import os 
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict

from Build_Model import CNNet, check_device


def predict_label (image_path, model, transform) :
    
    device = check_device()
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad() :
        
        output = model(image)
        
        predicted_label = torch.argmax(output, dim=1).item()

        '''
        # For showing probability
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][predicted_label].item()


    return predicted_label, confidence
    '''
    return predicted_label


if (__name__ == "__main__") :

    # Step 1. Load model.
    model = CNNet()

    model.load_state_dict(torch.load("../data/model.pth"))


    # Step 2. Predict.
    transform = transforms.Compose([
        transforms.Resize((201, 81)),
        transforms.ToTensor()
    ])

    test_image = "../data/preprocessed/spectrogram/test"

    # class_map = {'fake': 0, 'real': 1}
    inverse_class_map = {v: k for k, v in class_map.items()}
    # print(inverse_class_map)

    result = []
    
    for file_name in os.listdir(test_image) :
        if file_name.endswith('.png') : 
            image_path = os.path.join(test_image, file_name)
            # predicted_label, confidence = predict_label(image_path, model, transform)
            predicted_label = predict_label(image_path, model, transform)

            predicted_class = inverse_class_map[predicted_label]
            
            results.append((file_name, predicted_class))

    counts = defaultdict(lambda : {'real': 0, 'fake': 0})

    for filename, label in results :
        id = filename.split('_')[1]
        counts[id][label] += 1

    # Step 3. Write.        
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df.columns = ['id', 'fake', 'real']

    csv_data = []
    
    threshold = 0.3 # you can change. 

    for image_id, count_dict in counts.items() :
        real_count = count_dict['real']
        fake_count = count_dict['fake']
        
        cnt = real_count + fake_count
        hold = int(cnt * threshold)
        
        fake_count = max(fake_count - hold, 0)
        real_count = max(real_count - hold, 0)
        
        fake = (0.97 ** (fake_count)) * (0.03 ** (real_count))
        real = (0.97 ** (real_count)) * (0.03 ** (fake_count))

        
        csv_data.append({
            'id': "TEST_" + image_id,
            'fake': fake,
            'real': real
        })

    output_path = "../data/result.csv"

    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index = False)
