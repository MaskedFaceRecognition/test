// var countFiles = require('count-files')

// var stat = countFiles(dir, function(err, results) {
//     console.log('done counting')
//     console.log(results)
// })

// setInterval(function() {
//     console.log('current count', stats)
// }, 500)



// const result = countSync("../angry");
// console.log(result.fileCount);
// console.log(result.dirCount);

var countFiles = require('count-files')

var stats = countFiles(dir, function (err, results) {
  console.log('done counting')
  console.log(results) // { files: 10, dirs: 2, bytes: 234 }
})

setInterval(function () {
  console.log('current count', stats) // { files: 4, dirs: 1, bytes: 34 }
}, 500)