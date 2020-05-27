# backend_math_app

## Procesos 

En el archivo main "app" se inicializa flask, así mismo se mismo se inicializa la clase knn, después se setea el número de k's 

En el endpoint primero se espera que el json tenga el string de la imagen después se decodifica el string a un bitmap. después en el servidor se crea un archivo donde está la imagen para que el open cv pueda leer la imagen.


Se manda un JSON que contiene la base64 de la imagen y las dimensiones de la pantalla en la cual se está jugando, después se cambian los colores a blanco y negro, posteriormente se invierten los colores y se encuentra los contornos para identificar las imágenes por separado.


Con base a eso las imágenes se recortan, se centran y se hace un resize de 28px y con base a eso encuentra la distancia euclidiana de los 50k del dataset menos la imagen obtenida, se calculan lo k más cercanos y se obtiene el numero predicho.

Después se concatenan los resultados si el resultado es de dos o más dígitos, y regresa el resultado en el return.


## Contribuciones ✨

Damos nuestra palabra que hemos realizado esta actividad con integridad académica.
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
	<tr>
    	<td align="center"><a href="https://github.com/Poetickz"><img src="https://avatars1.githubusercontent.com/u/32653902?s=400&u=c9b802f74c9f0f36a653123865ddfd44d2c9308d&v=4" width="100px;" alt=""/><br /><sub><b>Alan Rocha</b></sub></a><br/><a href="#" title="Code">💻</a></td>
    	<td align="center"><a href="https://github.com/gabri3l0"><img src="https://avatars3.githubusercontent.com/u/42877617?s=460&u=0c97e12afc3c99b9d721bd2553185569832eb2e2&v=4" width="100px;" alt=""/><br /><sub><b>Gabriel Soto</b></sub></a><br/><a href="#" title="Code">💻</a></td>
    	<td align="center"><a href="https://github.com/memosteve"><img src="https://scontent.frex1-1.fna.fbcdn.net/v/t1.0-9/81507839_3148041308544260_8629329535277465600_n.jpg?_nc_cat=104&_nc_sid=09cbfe&_nc_eui2=AeE50-_TU40HXQdChFt29IfPMcIsMBbCuzIxwiwwFsK7Mk1J0HubJf731FJgOXbLK_4AEisscah_olKFyd8HEKSK&_nc_ohc=AG6V6yrRUX4AX-XUz5o&_nc_ht=scontent.frex1-1.fna&oh=f7293165c190785f73a6938da079884b&oe=5EF30CD6" width="100px;" alt=""/><br /><sub><b>Steve Albo</b></sub></a><br/><a href="#" title="Code">💻</a></td>
	</tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

