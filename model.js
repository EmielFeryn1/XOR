const setup = () => {
    let predict_button = document.getElementById("predict_button");
    predict_button.addEventListener("click", predict);
};

setup();

function predict()
{
  //read values here (let x = ...)
  postData(url="/predict", data={val1: 1, val2: 0})
  .then((data) => {
    console.log(data); // JSON data parsed by `data.json()` call
    document.getElementById("response").textContent = "val1: " + data["val1"] + " val2: " + data["val2"] + " result: " + data["result"];
  });
}
// Example POST method implementation:
async function postData(url = '', data = {}) {
  // Default options are marked with *
  const response = await fetch(url, {
    method: 'POST', // *GET, POST, PUT, DELETE, etc.
    mode: 'cors', // no-cors, *cors, same-origin
    cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
    credentials: 'same-origin', // include, *same-origin, omit
    headers: {
      'Content-Type': 'application/json'
      // 'Content-Type': 'application/x-www-form-urlencoded',
    },
    redirect: 'follow', // manual, *follow, error
    referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
    body: JSON.stringify(data), // body data type must match "Content-Type" header
  });
  return response.json(); // parses JSON response into native JavaScript objects
}