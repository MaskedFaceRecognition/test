function angryNum(){
    const fs = require('fs');
    const dir = './angry';

    fs.readdir(dir, (err, files) => {
        console.log(files.length);
        document.write(files.length);
    });
}

function happyNum(){
    const fs = require('fs');
    const dir = './happy';
    var count = 0

    fs.readdir(dir, (err, files) => {
        console.log(files.length);
        document.write(files.length);
    });
}

function neutralNum(){
    const fs = require('fs');
    const dir = './neutral';

    fs.readdir(dir, (err, files) => {
        console.log(files.length);
        document.write(files.length);
    });
}