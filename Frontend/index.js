function validate_user_input(url){
    url = url.trim();
    if (url === "") {
        return false;
    }

    if (!url.startsWith("https://")) {
        return false;
    }

    if (url.length <10 ) {
        return false;
    }

    return true;
}

function showToast(message) {
    let toast = document.getElementById("toast");
    toast.innerText = message;
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 5000);
}


let get_sentiment_btn = document.getElementById("get-sentiment")

get_sentiment_btn.addEventListener('click', (event)=>{
    let video_url = document.getElementById("video_url").value
    console.log(video_url);
    let url_valid = validate_user_input(video_url);
    if(url_valid===false){
        showToast("Invalid URL!!!")
    }
    else{
        //fetch the results
        fetch("https://sentiment-analyzer-5w4g.onrender.com/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
               yt_url : video_url
            })
        })
        .then(response => response.json())  // Parse JSON response
        .then(data => console.log("Success:", data))
        .catch(error => console.error("Error:", error));
        
    }
})