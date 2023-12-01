var know = {
    "hi": "Hi, I'm ChatBot. I'm here to help you",
    "hey": "Hey, I'm ChatBot. I'm here to help you",
    "hello": "Hello, I'm ChatBot. I'm here to help you",
    "hola": "Hola, I'm ChatBot. I'm here to help you",
    "hui": "Hui Hui, I'm ChatBot. I'm here to help you",
    "namaste": "Namaste, I'm ChatBot. I'm here to help you",
    "Or ji": "Han ji, I'm ChatBot. I'm here to help you",
    "hi chatbot": "Hi User, I'm here to help you",
    "hello chatbot": "Hello User, I'm here to help you",
    "hey chatbot": "Hey User, I'm here to help you",
    "hola chatbot": "Hola, I'm here to help you",
    "namaste chatbot": "Namaste, I'm here to help you",
    "what is your name": "I still don't have any name you can call me whatever you want",
    "chutiya kaisa hai": "that's rude :/",
    "how travelmate will help me": "<a href='../works.html'>Click here</a> to know about it",
    "ok":"Yes",
    "okay":"Yes",
    "thank you": "Welcome",
    "thanks": "Welcome",
    "dhanaywad": "Welcome",
    "thanks for helping": "Welcome",
    "thanks for your help": "Welcome",
    "i need help" : "yes please tell me i'm here",
    "is it really work" : "yes it is real and it will work"
};


function talk() {
    if (event.keyCode === 13) {
        var today = new Date();
        var time = today.getHours() + ":" + today.getMinutes();
        var element = document.getElementById("userSide");
        element.classList.remove("hidden");
        var user = document.getElementById("userBox").value;
        document.getElementById("userLog").innerHTML = user;
        document.getElementById("userBox").value = "";
        document.getElementById("chatLog").innerHTML = user + "<br>";
        document.getElementById("time1").innerHTML = time;
        document.getElementById("time2").innerHTML = time;
        var usermsg = user.toLowerCase();
        console.log(usermsg);
        if (usermsg in know) {
            document.getElementById("chatLog").innerHTML = know[usermsg] + "<br>";
        } else {
            document.getElementById("chatLog").innerHTML = "sorry, I don't know about that, I'm still learning" + "<br>";
        }
    } // if ends here
}

function talkwithbutton() {
    var today = new Date();
    var time = today.getHours() + ":" + today.getMinutes();
    var element2 = document.getElementById("userSide");
    element2.classList.remove("hidden");
    var user2 = document.getElementById("userBox").value;
    document.getElementById("userLog").innerHTML = user2;
    document.getElementById("chatLog").innerHTML = user2 + "<br>";
    document.getElementById("userBox").value = "";
    document.getElementById("time1").innerHTML = time;
    document.getElementById("time2").innerHTML = time;
    var usermsg2 = user2.toLowerCase();
    if (usermsg2 in know) {
        document.getElementById("chatLog").innerHTML = know[usermsg2] + "<br>";
    } else {
        document.getElementById("chatLog").innerHTML = "sorry, I don't know about that" + "<br>";
    }
}
