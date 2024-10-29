const vocabulary = {
  free: 1,
  entry: 2,
  contest: 3,
  call: 4,
  congratulations: 5,
  won: 6,
  meeting: 7,
  tomorrow: 8,
};

function preprocessText(text) {
  if (text) {
    const maxLength = 20; 
    const tokens = text.toLowerCase().split(' '); 
    const tokenIds = tokens.map(token => vocabulary[token] || 0);
    if (tokenIds.length > maxLength) {
      return tokenIds.slice(0, maxLength); 
    } else {
      return [...tokenIds, ...Array(maxLength - tokenIds.length).fill(0)];
    }
  } else {
    return Array(20).fill(0); 
  }
}

async function createModel(dataset) {
  const model = tf.sequential();
  
  model.add(tf.layers.embedding({ inputDim: 10000, outputDim: 128, inputLength: 20 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  const xTrainData = dataset.map(d => preprocessText(d.text));  
  const yTrainData = dataset.map(d => [d.label]);

  console.log('xTrainData:', xTrainData);
  console.log('Number of samples:', xTrainData.length);
  console.log('Shape of each sample:', xTrainData[0].length); 

  const xTrain = tf.tensor2d(xTrainData, [xTrainData.length, 20]);
  const yTrain = tf.tensor2d(yTrainData, [yTrainData.length, 1]);

  await model.fit(xTrain, yTrain, {
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss}`),
    },
  });

  return model;
}

let model;
(async () => {
  model = await createModel(dataset);
})();

document.getElementById('checkButton').addEventListener('click', async () => {
  if (!model) {
    alert('Model is still loading, please wait...');
    return;
  }
  const inputText = document.getElementById('messageInput').value;
  const processedInput = preprocessText(inputText);

  const inputTensor = tf.tensor2d([processedInput], [1, processedInput.length]);
  const prediction = model.predict(inputTensor).dataSync()[0];

  const resultText = prediction > 0.5 ? 'Spam' : 'Not Spam';
  document.getElementById('result').innerText = `Result: ${resultText}`;
});
