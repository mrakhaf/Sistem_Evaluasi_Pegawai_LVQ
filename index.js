var bobot_akhir = []

function training(){
    fetch('http://127.0.0.1:5000/train')
        .then(response => {
            // handle the response
            return response.text()
        })
        .then(data => {
            let a = JSON.parse(data)
            document.getElementById("b1").value = a[0][0]
            document.getElementById("b2").value = a[0][1]
            document.getElementById("b3").value = a[0][2]
        })
        .catch(error => {
            // handle the error
            console.log(error)
        });
}

const btnSubmit = document.getElementById("btn-testing")
btnSubmit.addEventListener("click", function (e) {
    e.preventDefault()
    testing()
})


function testing(){
    var formData = new FormData()
    formData.append("Nama Pegawai", document.getElementById("nama_pegawai").value)
    formData.append("Masa Kerja(thn)", document.getElementById("mk").value)
    formData.append("Usia", document.getElementById("usia").value)
    formData.append("Nilai Pelatihan", document.getElementById("np").value)
    formData.append("Nilai Kinerja", document.getElementById("nk").value)
    formData.append("Bobot Akhir", bobot_akhir)

    // fetch('http://127.0.0.1:5000/test', {
    // method: 'post',
    // headers: {
    //     "Content-Type": "application/json"
    // },
    // body : formData,
    // mode: 'no-cors'
    // })
    //     .then(response => {
    //         // handle the response
    //         console.log(response.text())
    //         return response.text()
    //     })
    //     .then(data => {
    //         console.log("masuk")
    //         console.log(data)
    //     })
    //     .catch(error => {
    //         // handle the error
    //         console.log(error)
    //     });

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://127.0.0.1:5000/test', true);
    xhr.onload = function () {
        data = JSON.parse(this.responseText)
        document.getElementById("hasilNama").value = data[0]['nama']
        document.getElementById("hasilEvaluasi").value = data[0]['hasil']
    };
    xhr.send(formData);
}





