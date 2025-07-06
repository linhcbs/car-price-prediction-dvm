let predict_button = document.querySelector("#predict-button");
let output_field = document.querySelector("#output-field");
let input_section = document.querySelector("#input-section");
let CATEGORICAL_CHOICES;

// feature name - feature text - feature type - isCategorical

let schema = [
  ["age", "Car age (years):", "number", false],
  ["runned-miles", "Runned miles (miles)", "number", false],
  ["engine-size", "Engine size (liters)", "number", false],
  ["engine-power", "Engine power (horsepower)", "number", false],
  ["width", "Width (milimeters)", "number", false],
  ["length", "Length (milimeters)", "number", false],
  ["average-mpg", "Average miles per gallon (mpg)", "number", false],
  ["top-speed", "Top-speed (mph)", "number", false],
  ["seat-num", "Number of seats", "number", false],
  ["maker", "Maker", "string", true],
  ["genmodel", "Genmodel", "string", true],
  ["bodytype", "Bodytype", "string", true],
  ["gearbox", "Type of gearbox", "string", true],
  ["fuel-type", "Fuel type", "string", true],
];


function formatNumberWithCommas(number) {
  return number.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function roundTo(num, decimalPlaces) {
  const factor = Math.pow(10, decimalPlaces);
  return Math.round(num * factor) / factor;
}


async function fetchData(url) {
  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Thử parse JSON
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      return await response.json();
    } else {
      return await response.text(); // fallback nếu không phải JSON
    }
  } catch (error) {
    console.error("Fetch error:", error);
    return null;
  }
}

async function fetchCatData() {
  try {
    const data = await fetchData("/categorical-choices");
    console.log("Nhận dữ liệu thành công");
    return data["choices"];
  } catch (error) {
    console.error("Lỗi khi lấy dữ liệu:", error);
    return null;
  }
}

async function getCategoricalChoices(featname) {
  if (CATEGORICAL_CHOICES == undefined) {
    CATEGORICAL_CHOICES = await fetchCatData();
  }

  return CATEGORICAL_CHOICES[featname];
}

async function renderFeatureInputs() {
  for (feature of schema) {
    let feature_div = null;
    let feature_name = feature[0];
    let feature_text = feature[1];
    let feature_type = feature[2];
    let feature_isCat = feature[3];

    let feature_input, feature_label;

    // feat div
    feature_div = document.createElement("div");
    feature_div.setAttribute("class", "feature-div");

    // feat label
    feature_label = document.createElement("label");
    feature_label.setAttribute("class", "feature-label");
    feature_label.setAttribute("for", "feature__" + feature_name);
    feature_label.textContent = feature_text;
    feature_div.appendChild(feature_label);

    // feat input
    if (feature_isCat == false) {
      feature_input = document.createElement("input");
      feature_input.setAttribute("type", "number");
      feature_input.setAttribute("min", 0);
      feature_input.setAttribute("class", "feature-input");
      feature_input.setAttribute("id", "feature__" + feature_name);
      feature_input.setAttribute("placeholder", "e.g: 1.0 ; 100 ; 1000");
    } else if (feature_isCat == true) {
      feature_input = document.createElement("select");
      feature_input.setAttribute("id", "feature__" + feature_name);
      feature_input.setAttribute("class", "feature-input");
      let choices = await getCategoricalChoices(feature_name);
      if (feature_name != "genmodel") {
        for (let choice of choices) {
          let option = document.createElement("option");
          option.textContent = choice;
          option.value = choice;

          if (choice == "Other") option.selected = true;
          feature_input.appendChild(option);
        }
      } else {
        let option = document.createElement("option");
        option.textContent = "Choose maker first";
        option.value = "Other";
        option.selected = true;
        option.disabled = true;
        feature_input.appendChild(option);
      }
    }
    feature_div.appendChild(feature_input);
    input_section.appendChild(feature_div);
  }
}

async function updateGenmodel(maker) {
  let choices = CATEGORICAL_CHOICES["genmodel"][maker];
  let genmodelInp = document.getElementById("feature__genmodel");

  genmodelInp.innerHTML = "";
  for (let choice of choices) {
    let option = document.createElement("option");
    option.textContent = choice;
    option.value = choice;

    if (choice == "Other") option.selected = true;
    genmodelInp.appendChild(option);
  }
}

function gatherFeatureInformation() {
  let data = {};

  for (let feat of schema) {
    let featname = feat[0];
    let feattype = feat[2];
    let featvalue = document.getElementById("feature__" + featname).value;

    try {
      if (feattype == "number") {
        if (featvalue == ''){
            featvalue = 0
        }
        featvalue = parseFloat(featvalue);
        
      }
      data[featname.replace('-', '_')] = featvalue;
    } catch {
        alert("Input không hợp lệ");
    }
  }
  return data;
}

async function sendPredictionRequest() {
    const payload = gatherFeatureInformation();
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Lỗi HTTP ${response.status}: ${errText}`);
        }

        const result = await response.json();
        showPrediction(result.prediction)
        console.log("Predicted result:", result);

    } catch (error) {
        console.error("Error sending request:", error);
        alert("Lỗi khi gửi dự đoán")

    }
}


function showPrediction (result){
    let field = document.querySelector("#output-field h2")
    
    field.textContent = formatNumberWithCommas(roundTo(result, 2))
}

async function enableEventListeners() {
  // genmodel changes according too maker
  let makerInp = document.getElementById("feature__maker");
  makerInp.addEventListener("change", (e) => {
    maker = e.target.value;
    updateGenmodel(maker);
  });

  // send for prediction
  let predictBtn = document.getElementById("predict-button");
  predictBtn.addEventListener("click", (e) => {
    sendPredictionRequest();
  });
}

window.onload = async () => {
  CATEGORICAL_CHOICES = await fetchCatData();
  await renderFeatureInputs();

  await enableEventListeners();
};
