//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2014-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni (printf is fajlmuvelet!)
// - new operatort hivni az onInitialization fÃ¼ggvÃ©nyt kivÃ©ve, a lefoglalt adat korrekt felszabadÃ­tÃ¡sa nÃ©lkÃ¼l
// - felesleges programsorokat a beadott programban hagyni
// - tovabbi kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan gl/glu/glut fuggvenyek hasznalhatok, amelyek
// 1. Az oran a feladatkiadasig elhangzottak ES (logikai AND muvelet)
// 2. Az alabbi listaban szerepelnek:
// Rendering pass: glBegin, glVertex[2|3]f, glColor3f, glNormal3f, glTexCoord2f, glEnd, glDrawPixels
// Transzformaciok: glViewport, glMatrixMode, glLoadIdentity, glMultMatrixf, gluOrtho2D,
// glTranslatef, glRotatef, glScalef, gluLookAt, gluPerspective, glPushMatrix, glPopMatrix,
// Illuminacio: glMaterialfv, glMaterialfv, glMaterialf, glLightfv
// Texturazas: glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glTexEnvi,
// Pipeline vezerles: glShadeModel, glEnable/Disable a kovetkezokre:
// GL_LIGHTING, GL_NORMALIZE, GL_DEPTH_TEST, GL_CULL_FACE, GL_TEXTURE_2D, GL_BLEND, GL_LIGHT[0..7]
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Tóth Gellért
// Neptun : QGZ6DV
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

//--------------------------------------------------------
// Heterogen collection
//--------------------------------------------------------
int intersect_counter = 0;

template<class T, int size = 10>
class Array {
	T *array[size];
	int numOfElements;
	//Array(const Array&);
	//Array& operator=(const Array &);

public:

	Array() :
			numOfElements(0) {
	}

	int SizeOf() {
		return numOfElements;
	}

	void Add(T *t) {
		if (numOfElements >= size) {
			exit(-1);
		}

		array[numOfElements++] = t;

	}

	T& operator[](int i) {
		if (i >= size)
			exit(-1);
		return *array[i];
	}
};

//--------------------------------------------------------
// Marix
//--------------------------------------------------------

void getMinor(float** source, float** destination, int row, int col,
		int order) {
	int colCount = 0;
	int rowCount = 0;

	for (int i = 0; i < order; i++) {
		if (i != row) {
			colCount = 0;
			for (int j = 0; j < order; j++) {
				if (j != col) {
					destination[rowCount][colCount] = source[i][j];
					colCount++;
				}
			}
			rowCount++;
		}

	}
}

float calcDeterminant(float** matrix, int order) {
	if (order == 1) {
		return matrix[0][0];
	}

	float det = 0;

	float** minor = new float*[order - 1];
	for (int i = 0; i < order - 1; i++) {
		minor[i] = new float[order - 1];
	}

	for (int i = 0; i < order; i++) {
		getMinor(matrix, minor, i, 0, order);

		det += (i % 2 == 1 ? -1 : 1) * matrix[i][0]
				* calcDeterminant(minor, order - 1);
	}

	for (int i = 0; i < order - 1; i++)
		delete[] minor[i];
	delete[] minor;

	return det;
}

struct myMatrix {
//	A B C D
//	E F G H
//	I J K L
//	M N O P

	float M[4][4];

	myMatrix() {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				M[i][j] = 0;
			}
		}
	}

	myMatrix(float a, float b, float c, float d, float e, float f, float g,
			float h, float i, float j) {
		M[0][0] = a;
		M[0][1] = b;
		M[0][2] = c;
		M[0][3] = d;
		M[1][0] = b;
		M[1][1] = e;
		M[1][2] = f;
		M[1][3] = g;
		M[2][0] = c;
		M[2][1] = f;
		M[2][2] = h;
		M[2][3] = i;
		M[3][0] = d;
		M[3][1] = g;
		M[3][2] = i;
		M[3][3] = j;
	}

	myMatrix(float a, float b, float c, float d, float e, float f, float g,
			float h, float i, float j, float k, float l, float m, float n,
			float o, float p) {
		M[0][0] = a;
		M[0][1] = b;
		M[0][2] = c;
		M[0][3] = d;
		M[1][0] = e;
		M[1][1] = f;
		M[1][2] = g;
		M[1][3] = h;
		M[2][0] = i;
		M[2][1] = j;
		M[2][2] = k;
		M[2][3] = l;
		M[3][0] = m;
		M[3][1] = n;
		M[3][2] = o;
		M[3][3] = p;
	}

	myMatrix operator+(const myMatrix & A) {
		myMatrix B;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				B.M[i][j] = M[i][j] + A.M[i][j];
			}
		}

		return B;
	}

	myMatrix operator*(float f) {
		myMatrix B;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				B.M[i][j] = M[i][j] * f;
			}
		}

		return B;
	}

	myMatrix operator*(const myMatrix & A) {
		myMatrix B;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				B.M[i][j] = 0;
				for (int k = 0; k < 4; k++) {
					B.M[i][j] = B.M[i][j] + M[i][k] * A.M[k][j];
				}

			}
		}

		return B;

	}

//	void printM() {
//		for (int i = 0; i < 4; i++) {
//			{
//				cout << M[i][0] << " " << M[i][1] << " " << M[i][2] << " "
//						<< M[i][3] << endl;
//			}
//
//		}
//
//	}

	myMatrix Transp() {

		myMatrix A;
		A.M[0][0] = M[0][0];
		A.M[0][1] = M[1][0];
		A.M[0][2] = M[2][0];
		A.M[0][3] = M[3][0];
		A.M[1][0] = M[0][1];
		A.M[1][1] = M[1][1];
		A.M[1][2] = M[2][1];
		A.M[1][3] = M[3][1];
		A.M[2][0] = M[0][2];
		A.M[2][1] = M[1][2];
		A.M[2][2] = M[2][2];
		A.M[2][3] = M[3][2];
		A.M[3][0] = M[0][3];
		A.M[3][1] = M[1][3];
		A.M[3][2] = M[2][3];
		A.M[3][3] = M[3][3];

		return A;
	}

	myMatrix inverse() {
		float** matrix = new float*[4];
		for (int i = 0; i < 4; i++)
			matrix[i] = new float[4];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				matrix[i][j] = M[i][j];
			}
		}

		myMatrix inverse = myMatrix();

		float** minor = new float*[3];
		for (int i = 0; i < 3; i++)
			minor[i] = new float[3];

		float determinant = calcDeterminant(matrix, 4);
		if (determinant == 0)
			return inverse;

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				getMinor(matrix, minor, j, i, 4);
				inverse.M[i][j] = calcDeterminant(minor, 3) / determinant;
				if ((i + j) % 2 == 1) {
					inverse.M[i][j] *= (-1);

				}
			}
		}

		delete[] matrix;
		delete[] minor;
		return inverse;
	}

};

//--------------------------------------------------------
// Vector in homogeneus coordinates
//--------------------------------------------------------

struct Vector {
	float x, y, z, w;

	Vector() {
		x = y = z = w = 0;
	}
	Vector(float x0, float y0, float z0 = 0, float w0 = 1) {
		x = x0;
		y = y0;
		z = z0;
		w = w0;
	}
	Vector operator*(float a) {
		return Vector(x * a, y * a, z * a);
	}
	Vector operator*(float a) const {
		return Vector(x * a, y * a, z * a);
	}
	Vector operator/(float a) {
		return Vector(x / a, y / a, z / a);
	}
	Vector operator+(const Vector& v) {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	Vector operator+(const Vector& v) const {
		return Vector(x + v.x, y + v.y, z + v.z);
	}
	Vector operator-(const Vector& v) {
		return Vector(x - v.x, y - v.y, z - v.z, 0);
	}
	Vector operator-(const Vector& v) const {
		return Vector(x - v.x, y - v.y, z - v.z, 0);
	}
	float operator*(const Vector& v) { 	// dot product
		return (x * v.x + y * v.y + z * v.z + w * v.w);
	}
	float operator*(const Vector& v) const { 	// dot product
		return (x * v.x + y * v.y + z * v.z);
	}

	Vector operator%(const Vector& v) { 	// cross product
		return Vector(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x,
				0);
	}
	float Length() {
		return sqrt(x * x + y * y + z * z);
	}
	Vector operator+(float a) {
		return Vector(x + a, y + a, z + a);
	}

	void Normalize() {
		float l = Length();
		if (l < 0.000001f) {
			x = 0;
			y = 0;
			z = 0;
		} else {
			x /= l;
			y /= l;
			z /= l;
			w = 0;
		}
	}

	void homogen() {
		if (w != 0) {
			x /= fabs(w);
			y /= fabs(w);
			z /= fabs(w);
			w = 0;

		}
	}

	Vector operator-(void) {
		return Vector(-x, -y, -z);
	}
	void operator+=(const Vector& v) {
		x += v.x;
		y += v.y;
		z += v.z;
	}
	void operator-=(const Vector& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
	}
	void operator*=(float f) {
		x *= f;
		y *= f;
		z *= f;
	}

	void printOut() {
		cout << "x:" << x << " y:" << y << " z:" << z << " w:" << w << endl;
	}

	Vector operator*(const myMatrix & A) {
		float V_a[4] = { 0, 0, 0, 0 };
		float V_b[4] = { x, y, z, w };

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++)
				V_a[j] += ((V_b[i] * (A.M[i][j])));
		}

		Vector V = Vector(V_a[0], V_a[1], V_a[2], V_a[3]);

		return V;
	}

	Vector operator*(const myMatrix & A) const {
		float V_a[4] = { 0, 0, 0, 0 };
		float V_b[4] = { x, y, z, w };

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++)
				V_a[j] += ((V_b[i] * (A.M[i][j])));
		}

		Vector V = Vector(V_a[0], V_a[1], V_a[2], V_a[3]);

		return V;
	}

};

//--------------------------------------------------------
// Spektrum illetve szin
//--------------------------------------------------------
struct Color {
	float r, g, b;

	Color() {
		r = g = b = 0;
	}
	Color(float r0, float g0, float b0) {
		r = r0;
		g = g0;
		b = b0;
	}
	Color operator*(float a) {
		return Color(r * a, g * a, b * a);
	}
	Color operator*(const Color& c) {
		return Color(r * c.r, g * c.g, b * c.b);
	}
	Color operator+(const Color& c) {
		return Color(r + c.r, g + c.g, b + c.b);
	}
	Color operator-(const Color& c) {
		return Color(r - c.r, g - c.g, b - c.b);
	}
	Color operator/(const Color& c) {
		return Color(r / c.r, g / c.g, b / c.b);
	}
	Color operator/(float c) {
		return Color(r / c, g / c, b / c);
	}
	void operator+=(const Color& c) {
		r += c.r;
		g += c.g;
		b += c.b;
	}
};

Color BLACK = Color(0.0f, 0.0f, 0.0f);
Color WHITE = Color(1.0f, 1.0f, 1.0f);
const int screenWidth = 600;	// alkalmazÃ¡s ablak felbontÃ¡sa
const int screenHeight = 600;
Color image[screenWidth * screenHeight];	// egy alkalmazÃ¡s ablaknyi kÃ©p
float EPSILON = 0.001;
int MAX_DEPTH = 3;
float PI = 3.14159265359;

//--------------------------------------------------------
// Material with every proterties
//--------------------------------------------------------

class Material {
public:
	Color ka;
	Color ks;
	Color kd;
	Color n;
	Color k;
	Color F0;
	float shininess;
	bool isReflective;
	bool isRefractive;

	Material() {
		ka = ks = kd = n = k = F0 = WHITE;
		isReflective = isRefractive = false;
		shininess = 0;
	}

	Material(Color ks, Color kd, Color n, Color k, Color ka, float shine,
			bool isReflect, bool isRefract) {
		this->ka = ka;

		this->ks = ks; //SetSpecColor(ks);

		this->kd = kd; //SetDiffColor(kd);
		this->n = n;
		this->k = k;
		this->shininess = shine;
		this->isReflective = isReflect;
		this->isRefractive = isRefract;

	}

	bool isIsReflective() const {
		return isReflective;
	}

	void setIsReflective(bool isReflective = false) {
		this->isReflective = isReflective;
	}

	bool isIsRefractive() const {
		return isRefractive;
	}

	void setIsRefractive(bool isRefractive = false) {
		this->isRefractive = isRefractive;
	}

	const Color& getK() const {
		return k;
	}

	void setK(const Color& k) {
		this->k = k;
	}

	const Color& getKa() const {
		return ka;
	}

	void setKa(const Color& ka) {
		this->ka = ka;
	}

	const Color& getKd() const {
		return kd;
	}

	void setKd(const Color& kd) {
		this->kd = kd;
	}

	const Color& getKs() const {
		return ks;
	}

	void setKs(const Color& ks) {
		this->ks = ks;
	}

	const Color& getN() const {
		return n;
	}

	void setN(const Color& n) {
		this->n = n;
	}

	float getShine() const {
		return shininess;
	}

	void setShine(float shine = 0) {
		this->shininess = shine;
	}

	void calcF0() {
		F0 = ((n - WHITE) * (n - WHITE) + k * k)
				/ ((n + WHITE) * (n + WHITE) + k * k);
	}

	Color Fresnel(Vector& inDirection, Vector& normalVector) {
		inDirection.Normalize();
		normalVector.Normalize();
		float cosA = fabs(normalVector * inDirection);
		calcF0();
		return F0 + (WHITE - F0) * pow(1 - cosA, 5);

	}

	Vector reflect(Vector& inDirection, Vector& normalVector) {
		Vector inDir = inDirection;
		inDir.Normalize();
		Vector normVect = normalVector;
		normVect.Normalize();
//		cout<<"Dir"<<endl;
//				inDir.printOut();
//				cout<<"Norm"<<endl;
//				normVect.printOut();
//
//		float cosA = fabs(normVect * inDir);
//		cout<<cosA<<endl;
//		cout<<endl;

		return inDir - normVect * (normVect * inDir) * 2;
	}

	Vector refract(Vector& inDirection, Vector& normalVector) {
		inDirection.Normalize();
		normalVector.Normalize();
		Vector reflacted = Vector();
		float n[3] = { this->n.r, this->n.g, this->n.b };

		float N = this->n.r;
		Vector myNormalVector = normalVector;
		float cosA = fabs(myNormalVector * inDirection);
		if (cosA <= 0) {
			cosA = -cosA;
			myNormalVector = normalVector * (-1);
			N = 1 / this->n.r;

		}
		float disc = 1 - (1 - cosA * cosA) / N / N;


		if (disc < 0) {
			return reflect(inDirection, myNormalVector);
		}
		reflacted = inDirection / N + myNormalVector * (cosA / N - sqrt(disc));
		return reflacted;
	}

	Color shade_BRDF(Vector normalVector, Vector viewDirection,
			Vector lightDirection, Color lightIntense) { // diffuz es spekularis tag is benne van.
		Color reflected = Color();

		normalVector.Normalize();
		viewDirection.Normalize();
		lightDirection.Normalize();

		float cosTheta = normalVector * lightDirection;
		if (cosTheta < 0)
			return reflected;
		reflected = lightIntense * kd * cosTheta;
		Vector halfWay = (viewDirection + lightDirection);
		halfWay.Normalize();
		float cosDelta = normalVector * halfWay;
		if (cosDelta < 0)
			return reflected;
		return reflected + lightIntense * ks * pow(cosDelta, shininess);

	}

};

class Gold: Material {
	Gold() :
			Material() {
	}
	;
};

class Silver: Material {
	Silver() :
			Material() {
	}
	;
};

//--------------------------------------------------------
// Intersection point
//--------------------------------------------------------

struct Hit {
public:
	float t;
	Vector intersectPoint; // metszespont
	Vector normalVector;
	Material* material;

	Hit() {
		t = -1;
		intersectPoint = normalVector = Vector();
		material = new Material();
	}

	void operator =(const Hit h) {
		t = h.t;
		intersectPoint = h.intersectPoint;
		normalVector = h.normalVector;
		material = h.material;
	}

//	~Hit() {
//		delete material;
//	}
};

class Ray {
public:
	Vector startPoin;
	Vector rayDirection;

	Ray(Vector start, Vector direction) {
		startPoin = start;
		rayDirection = direction;
	}
};

Vector matrix_vector_multi(myMatrix A, Vector v) {

	float V_a[4] = { 0, 0, 0, 0 };
	float V_b[4] = { v.x, v.y, v.z, v.w };

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++)
			V_a[i] += ((V_b[j] * (A.M[i][j])));
	}

	Vector V = Vector(V_a[0], V_a[1], V_a[2], V_a[3]);

	return V;

}

//--------------------------------------------------------
// Objects
//--------------------------------------------------------

class Intersectable {
public:
	Material* material;
	myMatrix quadric;
	myMatrix transfom;

	Intersectable() {
		material = new Material();
	}

	Intersectable(Material* m) {
		material = m;
		transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	virtual Hit intersect(const Ray& ray) {
//		Vector point = ray.startPoin;
//		Vector direction = ray.rayDirection;
//		direction.Normalize();
		myMatrix tr_inv = transfom.inverse();
		Vector point = ray.startPoin * tr_inv;
		Vector direction = ray.rayDirection * tr_inv;
//		direction.Normalize();
		Hit h = Hit();

		Vector a_11 = (direction * quadric);
		float a_1 = a_11 * direction;
		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

		double discriminant_1 = b_1 * b_1 - 4 * a_1 * c_1;

		if (discriminant_1 < 0)
			return h; // visszateres megadasa;

		float sqrt_discriminant_1 = sqrt(discriminant_1);
		//
		float t1 = (-b_1 + sqrt_discriminant_1) / 2 / a_1;
		float t2 = (-b_1 - sqrt_discriminant_1) / 2 / a_1;
		//
		if (t1 < EPSILON)
			t1 = -EPSILON;
		if (t2 < EPSILON)
			t2 = -EPSILON;
		if (t1 < 0 && t2 < 0)
			return h;

		float t;
		if (t1 < 0)
			return h;
		if (t2 > 0)
			t = t2;
		else
			t = t1;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;

		Vector intersectpoint_model = point + (direction * t);
		Vector transformed_back = intersectpoint_model * transfom;

		myMatrix tr_inv_transp = tr_inv.Transp();

		hit.intersectPoint = transformed_back;
//		hit.normalVector = calcNormalVector(hit.intersectPoint);
		hit.normalVector = calcNormalVector(intersectpoint_model)
				* tr_inv_transp;

		hit.normalVector.Normalize();
		intersect_counter++;

		return hit;
	}

	virtual Vector calcNormalVector(Vector intersectPoint) {

		float dx, dy, dz;

		dx = Vector(1, 0, 0, 0) * quadric * intersectPoint
				+ intersectPoint * quadric * Vector(1, 0, 0, 0);
		dy = Vector(0, 1, 0, 0) * quadric * intersectPoint
				+ intersectPoint * quadric * Vector(0, 1, 0, 0);
		dz = Vector(0, 0, 1, 0) * quadric * intersectPoint
				+ intersectPoint * quadric * Vector(0, 0, 1, 0);

		return Vector(dx, dy, dz, 0);

	}

	Material* getMaterial() {
		return material;
	}

	virtual void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

	virtual ~Intersectable() {
		delete material;
	}

};

class Sphere: Intersectable {
public:
//	myMatrix quadric;

	Sphere(Material* m) {
		material = m;
		quadric = myMatrix(4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1);
		transfom = myMatrix(0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, -0.2, 0.1,
				-0.8, 1);
	}

	~Sphere() {
	}

	void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

};

class Ellipsoid: Intersectable {
//	myMatrix quadric;
public:
	static int num_of_element;
	Vector y_axis;
//	Ellipsoid next_element;
	Ellipsoid(Material* m) {
		y_axis = Vector(0, 1, 0, 0);
		material = m;
		quadric = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1);

		transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

	}

	void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

	Hit intersect(const Ray& ray) {

		myMatrix tr_inv = transfom.inverse();
		Vector point = ray.startPoin * tr_inv;
		Vector direction = ray.rayDirection * tr_inv;
		Hit h = Hit();

		Vector a_11 = (direction * quadric);
		float a_1 = a_11 * direction;
		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

		double discriminant_1 = b_1 * b_1 - 4 * a_1 * c_1;

		if (discriminant_1 < 0)
			return h; // visszateres megadasa;

		float sqrt_discriminant_1 = sqrt(discriminant_1);
		//
		float t1 = (-b_1 + sqrt_discriminant_1) / 2 / a_1;
		float t2 = (-b_1 - sqrt_discriminant_1) / 2 / a_1;
		//
		if (t1 < EPSILON)
			t1 = -EPSILON;
		if (t2 < EPSILON)
			t2 = -EPSILON;
		if (t1 < 0 && t2 < 0)
			return h;

		float t;
		if (t1 < 0)
			return h;
		if (t2 > 0)
			t = t2;
		else
			t = t1;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;

		Vector intersectpoint_model = point + (direction * t);
		Vector transformed_back = intersectpoint_model * transfom;

		myMatrix tr_inv_transp = tr_inv.Transp();

		hit.intersectPoint = transformed_back;
		hit.normalVector = calcNormalVector(intersectpoint_model)
				* tr_inv_transp;

		hit.normalVector.Normalize();

//		if (hit.intersectPoint.y < 0.35 && hit.intersectPoint.y>0.3 ) {
////			cout << "Megvan:" << endl;
////			hit.intersectPoint.printOut();
//			return h;
//
//		}

//		if(fabs(hit.intersectPoint.y - (0.2))<0.001 && fabs(hit.intersectPoint.x - (-0.15))<0.001){
//			cout << "Megvan:" << endl;
//			cout << "Normal:" << endl;
//			hit.normalVector.printOut();
////			cout << "origo:" << endl;
////			hit.intersectPoint.printOut();
//			return h;
//
//		}

//		if(num_of_element<1)

		return hit;
	}

	myMatrix getTheNewCoordSys(Hit intersect_point) {
		Vector new_y_axis = intersect_point.normalVector;
		Vector new_z_axis = y_axis % intersect_point.normalVector;
		Vector new_x_axis = new_z_axis % new_y_axis;
		Vector new_origo = intersect_point.intersectPoint;

		myMatrix new_coord_sys = myMatrix(new_x_axis.x, new_x_axis.y,
				new_x_axis.z, 0, new_y_axis.x, new_y_axis.y, new_y_axis.z, 0,
				new_z_axis.x, new_z_axis.y, new_z_axis.z, 0, new_origo.x,
				new_origo.y, new_origo.z, 1);

		return new_coord_sys;

	}

};

int Ellipsoid::num_of_element = 0;

class Plane_XZ_Down: Intersectable {
public:
	Plane_XZ_Down(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0,
				0.5);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(0, 1, 0);
		return n;
	}

	Hit intersect(const Ray& ray) {
		Vector point = ray.startPoin;
		Vector direction = ray.rayDirection;
		direction.Normalize();
		Hit h = Hit();

		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

//				cout<<" b:"<<b_1<<" c_1:"<<c_1<<endl;

		float t = -c_1 / b_1;

		if (t < EPSILON)
			t = -EPSILON;

		if (t < 0)
			return h;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;
		hit.intersectPoint = ray.startPoin + (ray.rayDirection * t);
		hit.normalVector = calcNormalVector(hit.intersectPoint);
		hit.normalVector.Normalize();
		intersect_counter++;

		if (hit.intersectPoint.x < -0.51 || hit.intersectPoint.x > 0.51
				|| hit.intersectPoint.z < -0.51 || hit.intersectPoint.z > 0.51)
			return h;

		return hit;
	}

};

class Plane_XZ_Up: Intersectable {
public:
	Plane_XZ_Up(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0,
				-0.5);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(0, -1, 0);
		return n;
	}

	Hit intersect(const Ray& ray) {
		Vector point = ray.startPoin;
		Vector direction = ray.rayDirection;
		direction.Normalize();
		Hit h = Hit();

		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

//				cout<<" b:"<<b_1<<" c_1:"<<c_1<<endl;

		float t = -c_1 / b_1;

		if (t < EPSILON)
			t = -EPSILON;

		if (t < 0)
			return h;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;
		hit.intersectPoint = ray.startPoin + (ray.rayDirection * t);
		hit.normalVector = calcNormalVector(hit.intersectPoint);
		hit.normalVector.Normalize();
		intersect_counter++;

		if (hit.intersectPoint.x < -0.51 || hit.intersectPoint.x > 0.51
				|| hit.intersectPoint.z < -0.51 || hit.intersectPoint.z > 0.51)
			return h;

		return hit;
	}

};

class Plane_YZ_Right: Intersectable {
public:
	Plane_YZ_Right(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
				-0.5);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(-1, 0, 0);
		return n;
	}

	Hit intersect(const Ray& ray) {
		Vector point = ray.startPoin;
		Vector direction = ray.rayDirection;
		direction.Normalize();
		Hit h = Hit();

		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

//				cout<<" b:"<<b_1<<" c_1:"<<c_1<<endl;

		float t = -c_1 / b_1;

		if (t < EPSILON)
			t = -EPSILON;

		if (t < 0)
			return h;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;
		hit.intersectPoint = ray.startPoin + (ray.rayDirection * t);
		hit.normalVector = calcNormalVector(hit.intersectPoint);
		hit.normalVector.Normalize();
		intersect_counter++;

		if (hit.intersectPoint.y < -0.51 || hit.intersectPoint.y > 0.51
				|| hit.intersectPoint.z < -0.51 || hit.intersectPoint.z > 0.51)
			return h;

		return hit;
	}

};

class Plane_YZ_Left: Intersectable {
public:
	Plane_YZ_Left(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
				0.5);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(1, 0, 0);
		return n;
	}

	Hit intersect(const Ray& ray) {
		Vector point = ray.startPoin;
		Vector direction = ray.rayDirection;
		direction.Normalize();
		Hit h = Hit();

		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

//				cout<<" b:"<<b_1<<" c_1:"<<c_1<<endl;

		float t = -c_1 / b_1;

		if (t < EPSILON)
			t = -EPSILON;

		if (t < 0)
			return h;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;
		hit.intersectPoint = ray.startPoin + (ray.rayDirection * t);
		hit.normalVector = calcNormalVector(hit.intersectPoint);
		hit.normalVector.Normalize();
		intersect_counter++;

		if (hit.intersectPoint.y < -0.51 || hit.intersectPoint.y > 0.51
				|| hit.intersectPoint.z < -0.51 || hit.intersectPoint.z > 0.51)
			return h;

		return hit;
	}

};

class Plane_XY_Front: Intersectable {
public:
	Plane_XY_Front(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5,
				0.5);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(0, 0, -1);
		return n;
	}

	Hit intersect(const Ray& ray) {
		Vector point = ray.startPoin;
		Vector direction = ray.rayDirection;
		direction.Normalize();
		Hit h = Hit();

		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

//				cout<<" b:"<<b_1<<" c_1:"<<c_1<<endl;

		float t = -c_1 / b_1;

		if (t < EPSILON)
			t = -EPSILON;

		if (t < 0)
			return h;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;
		hit.intersectPoint = ray.startPoin + (ray.rayDirection * t);
		hit.normalVector = calcNormalVector(hit.intersectPoint);
		hit.normalVector.Normalize();
		intersect_counter++;

		if (hit.intersectPoint.y < -0.51 || hit.intersectPoint.y > 0.51
				|| hit.intersectPoint.x < -0.51 || hit.intersectPoint.x > 0.51)
			return h;

		return hit;
	}

};

class Plane_XY_Back: Intersectable {
public:
	Plane_XY_Back(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5,
				-0.5);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(0, 0, 1);
		return n;
	}

	Hit intersect(const Ray& ray) {
		Vector point = ray.startPoin;
		Vector direction = ray.rayDirection;
		direction.Normalize();
		Hit h = Hit();

		Vector b_11 = (point * quadric);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * quadric);
		float c_1 = c_11 * point;

//				cout<<" b:"<<b_1<<" c_1:"<<c_1<<endl;

		float t = -c_1 / b_1;

		if (t < EPSILON)
			t = -EPSILON;

		if (t < 0)
			return h;

		Hit hit = Hit();
		hit.material = material;
		hit.t = t;
		hit.intersectPoint = ray.startPoin + (ray.rayDirection * t);
		hit.normalVector = calcNormalVector(hit.intersectPoint);
		hit.normalVector.Normalize();
		intersect_counter++;

		if (hit.intersectPoint.y < -0.51 || hit.intersectPoint.y > 0.51
				|| hit.intersectPoint.x < -0.51 || hit.intersectPoint.x > 0.51)
			return h;

		return hit;
	}

};

class Plane_XZ: Intersectable {
public:
	Plane_XZ(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0,
				0);
		transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(0, 1, 0);
		return n;
	}


	bool intersectPoints(Vector point, Vector direction, myMatrix m,
			float& t1) {

		Vector b_11 = (point * m);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * m);
		float c_1 = c_11 * point - 1;

		t1 = -c_1 / b_1;

		return true;
	}

	Hit intersect(const Ray& ray) {
		Hit h = Hit();

		float t1;

		myMatrix tr_inv = transfom.inverse();
		Vector point = ray.startPoin * tr_inv;
		Vector direction = ray.rayDirection * tr_inv;
		bool b = intersectPoints(point, direction, quadric, t1);

		if (!b)
			return h;

		if (t1 < EPSILON)
			t1 = -EPSILON;

		if (t1 < 0)
			return h;

		float t;

		Vector v1 = point + (direction * t1);

		myMatrix tr_inv_transp = tr_inv.Transp();
		Hit hit = Hit();
		hit.material = material;

		if (v1.x < -0.51 || v1.x > 0.51 || v1.z < -0.51 || v1.z > 0.51)
			return h;

		Vector transformed_back = v1 * transfom;

		hit.t = t1;
		hit.intersectPoint = transformed_back;
		hit.normalVector = calcNormalVector(v1) * tr_inv_transp;
		hit.normalVector.Normalize();

		return hit;
	}

	void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

};

class Plane_XY: Intersectable {
public:
	Plane_XY(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.5, 0);
		transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(0, 0, -1);
		return n;
	}

	bool intersectPoints(Vector point, Vector direction, myMatrix m,
			float& t1) {

		Vector b_11 = (point * m);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * m);
		float c_1 = c_11 * point - 1;

		t1 = -c_1 / b_1;

		return true;
	}

	Hit intersect(const Ray& ray) {
		Hit h = Hit();

		float t1;

		myMatrix tr_inv = transfom.inverse();
		Vector point = ray.startPoin * tr_inv;
		Vector direction = ray.rayDirection * tr_inv;
		bool b = intersectPoints(point, direction, quadric, t1);

		if (!b)
			return h;

		if (t1 < EPSILON)
			t1 = -EPSILON;

		if (t1 < 0)
			return h;

		float t;

		Vector v1 = point + (direction * t1);

		myMatrix tr_inv_transp = tr_inv.Transp();
		Hit hit = Hit();
		hit.material = material;

		if (v1.x < -0.51 || v1.x > 0.51 || v1.y < -0.51 || v1.y > 0.51)
			return h;

		Vector transformed_back = v1 * transfom;

		hit.t = t1;
		hit.intersectPoint = transformed_back;
		hit.normalVector = calcNormalVector(v1) * tr_inv_transp;
		hit.normalVector.Normalize();

		return hit;
	}

	void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

};

class Plane_YZ: Intersectable {
public:
	Plane_YZ(Material* m) {
		material = m;
		quadric = myMatrix(0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
				0);
		transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	Vector calcNormalVector(Vector intersectPoint) {
		Vector n = Vector(1, 0, 0);
		return n;
	}


	bool intersectPoints(Vector point, Vector direction, myMatrix m,
			float& t1) {

		Vector b_11 = (point * m);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * m);
		float c_1 = c_11 * point - 1;

		t1 = -c_1 / b_1;

		return true;
	}

	Hit intersect(const Ray& ray) {
		Hit h = Hit();

		float t1;

		myMatrix tr_inv = transfom.inverse();
		Vector point = ray.startPoin * tr_inv;
		Vector direction = ray.rayDirection * tr_inv;
		bool b = intersectPoints(point, direction, quadric, t1);

		if (!b)
			return h;

		if (t1 < EPSILON)
			t1 = -EPSILON;

		if (t1 < 0)
			return h;

		float t;

		Vector v1 = point + (direction * t1);

		myMatrix tr_inv_transp = tr_inv.Transp();
		Hit hit = Hit();
		hit.material = material;

		if (v1.y < -0.51 || v1.y > 0.51 || v1.z < -0.51 || v1.z > 0.51)
			return h;

		Vector transformed_back = v1 * transfom;

		hit.t = t1;
		hit.intersectPoint = transformed_back;
		hit.normalVector = calcNormalVector(v1) * tr_inv_transp;
		hit.normalVector.Normalize();

		return hit;
	}

	void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

};


class Cylinder: Intersectable {
//	myMatrix quadric;
public:
	Cylinder(Material* m) {
		material = m;

		quadric = myMatrix(6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, -1);
		transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

	}
	bool intersectPoints(Vector point, Vector direction, myMatrix m, float& t1,
			float& t2) {

		Vector a_11 = (direction * m);
		float a_1 = a_11 * direction;
		Vector b_11 = (point * m);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * m);
		float c_1 = c_11 * point - 1;

		double discriminant_1 = b_1 * b_1 - 4 * a_1 * c_1;

		if (discriminant_1 < 0)
			return false; // visszateres megadasa;

		float sqrt_discriminant_1 = sqrt(discriminant_1);

		t1 = (-b_1 + sqrt_discriminant_1) / 2 / a_1;
		t2 = (-b_1 - sqrt_discriminant_1) / 2 / a_1;

		return true;
	}

	Hit intersect(const Ray& ray) {
		Hit h = Hit();

		float t1;
		float t2;
		myMatrix tr_inv = transfom.inverse();
		Vector point = ray.startPoin * tr_inv;
		Vector direction = ray.rayDirection * tr_inv;
		bool b = intersectPoints(point, direction, quadric, t1, t2);

		if (!b)
			return h;

		if (t1 < EPSILON)
			t1 = -EPSILON;
		if (t2 < EPSILON)
			t2 = -EPSILON;
		if (t1 < 0 && t2 < 0)
			return h;

		if (t1 < 0)
			return h;

		Vector v1 = point + (direction * t1);
		Vector v2 = point + (direction * t2);

		myMatrix tr_inv_transp = tr_inv.Transp();
		Hit hit = Hit();
		hit.material = material;
		if (v2.y > 0.6 || v2.y < -1) {
			if (v1.y > 0.6 || v1.y < -1)
				return h;
			Vector transformed_back = v1 * transfom;

			hit.t = t1;
			hit.intersectPoint = transformed_back;
			hit.normalVector = calcNormalVector(v1) * tr_inv_transp;
			hit.normalVector.Normalize();
			hit.normalVector = hit.normalVector * (-1);

		} else if (v2.y >= -1) {
			Vector transformed_back = v2 * transfom;

			hit.t = t2;
			hit.intersectPoint = transformed_back;
			hit.normalVector = calcNormalVector(v2) * tr_inv_transp;
			hit.normalVector.Normalize();

		}

		return hit;
	}

	void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

};

class Paraboloid: Intersectable {
public:
	Paraboloid(Material* m) {
		material = m;
		quadric = myMatrix(1, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 1, 0, 0, -0.5, 0,
				0);
		transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	}

	bool intersectPoints(Vector point, Vector direction, myMatrix m, float& t1,
			float& t2) {

		Vector a_11 = (direction * m);
		float a_1 = a_11 * direction;
		Vector b_11 = (point * m);
		float b_1 = (b_11 * direction) * 2;
		Vector c_11 = (point * m);
		float c_1 = c_11 * point - 1;

		double discriminant_1 = b_1 * b_1 - 4 * a_1 * c_1;

		if (discriminant_1 < 0)
			return false; // visszateres megadasa;

		float sqrt_discriminant_1 = sqrt(discriminant_1);

		t1 = (-b_1 + sqrt_discriminant_1) / 2 / a_1;
		t2 = (-b_1 - sqrt_discriminant_1) / 2 / a_1;

		return true;
	}

	Hit intersect(const Ray& ray) {
		Hit h = Hit();

		float t1;
		float t2;
		myMatrix tr_inv = transfom.inverse();
		Vector point = ray.startPoin * tr_inv;
		Vector direction = ray.rayDirection * tr_inv;
		bool b = intersectPoints(point, direction, quadric, t1, t2);

		if (!b)
			return h;

		if (t1 < EPSILON)
			t1 = -EPSILON;
		if (t2 < EPSILON)
			t2 = -EPSILON;
		if (t1 < 0 && t2 < 0)
			return h;

		float t;
		if (t1 < 0)
			return h;

		Vector v1 = point + (direction * t1);
		Vector v2 = point + (direction * t2);

		myMatrix tr_inv_transp = tr_inv.Transp();
		Hit hit = Hit();
		hit.material = material;

		if (v2.y > 0.6) {
			if (v1.y > 0.6)
				return h;
			Vector transformed_back = v1 * transfom;

			hit.t = t1;
			hit.intersectPoint = transformed_back;
			hit.normalVector = calcNormalVector(v1) * tr_inv_transp;
			hit.normalVector.Normalize();
			hit.normalVector = hit.normalVector * (-1);

		} else if (v2.y >= -1) {
			Vector transformed_back = v2 * transfom;

			hit.t = t2;
			hit.intersectPoint = transformed_back;
			hit.normalVector = calcNormalVector(v2) * tr_inv_transp;
			hit.normalVector.Normalize();

		}

		return hit;
	}

	void setTrasformationMatrix(myMatrix transformation) {
		transfom = transformation;
	}

};

class MyCamera {
	Vector eyePosition;
	Vector lookAtPoint;
	Vector upDirection;
	Vector rightDirection;
	float resolutionX;
	float resolutionY;
public:
	MyCamera() {
		eyePosition = lookAtPoint = upDirection = rightDirection = Vector();
		resolutionX = resolutionY = 0;
	}

	MyCamera(Vector eye, Vector lookat, Vector up) {
		eyePosition = eye;
		lookAtPoint = lookat;
		up.Normalize();
		upDirection = up;

		Vector ahead = (lookAtPoint - eyePosition);
		ahead.Normalize();
		rightDirection = (ahead % upDirection);
		rightDirection.Normalize();
		resolutionX = 600;
		resolutionY = 600;
	}

	Ray GetRay(float x, float y) {
		Vector p = lookAtPoint + rightDirection * (2 * x / resolutionX - 1)
				+ upDirection * (2 * y / resolutionY - 1);
		Vector dir = p - eyePosition;
		dir.Normalize();
		Ray ray = Ray(eyePosition, dir);
		return ray;
	}
};

class LightSource {
public:
	Vector position;
	Color lightRadiation;

	LightSource(Vector pos, Color intens) {
		position = pos;
		lightRadiation = intens;
	}

	virtual Color getRadiation() {
		return lightRadiation;
	}

	virtual Vector lightDirection(Vector point) {
		Vector v = position - point;
		v.Normalize();
		return v;
	}

	virtual Color getRadiation(Vector point) {
		return lightRadiation / ((point - position) * (point - position));
	}

	virtual ~LightSource() {

	}

};

class DirectionalLightSource: LightSource {
public:
	Vector direction;

	DirectionalLightSource(Vector dir, Vector position, Color intense) :
			LightSource(position, intense) {
		this->direction = dir;
	}

	Vector lightDirection(Vector point) {

		return direction;
	}

};

class PositionLightSource: LightSource {
public:
	PositionLightSource(Vector pos, Color intens) :
			LightSource(pos, intens) {
	}

	Vector lightDirection(Vector point) {
		Vector v = position - point;
		v.Normalize();
		return v;
	}

//	Color getRadiation(Vector point) {
//		return lightRadiation / ((point - position) * (point - position));
//	}

//Color getRadiation(Vector point) {
//	return lightRadiation;
//}
};

class Scene {
	Array<Intersectable, 10> objects;
	Array<LightSource, 3> lightSources;
	MyCamera camera;
	Color ambientColor;
public:
	Scene() {
		objects = Array<Intersectable, 10>();
		lightSources = Array<LightSource, 3>();
		camera = MyCamera();
		ambientColor = Color();
	}

	Hit firstIntersect(Ray ray) {
		Hit hit = Hit();

		float t = -1;

		for (int i = 0; i < objects.SizeOf(); i++) {

			Hit testing = objects[i].intersect(ray);

			if (testing.t > 0 && (testing.t < t || t < 0)) {
				t = testing.t;
				hit = testing;
			}
		}
		return hit;
	}

	Color trace(Ray ray, int depth) {
		if (depth > MAX_DEPTH){
			return ambientColor;
		}
		Hit hit = firstIntersect(ray);
		if (hit.t < 0.000001){

			return ambientColor;
		}

		Color outRadiance = Color(0, 0, 0) + hit.material->kd * 0.3;

		for (int i = 0; i < lightSources.SizeOf(); i++) {

			Ray shadowRay = Ray(hit.intersectPoint + hit.normalVector * (0.001),
					lightSources[i].lightDirection(hit.intersectPoint)); //Sugar inditasa a metszespontbol a fenyforras fele
			Hit shadowHit = firstIntersect(shadowRay);
			Vector pointLightVector = lightSources[i].position
					- hit.intersectPoint;
			if (shadowHit.t < 0 || shadowHit.t > pointLightVector.Length()) { // amikor nincsen arnyek
				outRadiance += hit.material->shade_BRDF(hit.normalVector,
						-ray.rayDirection, pointLightVector,
						lightSources[i].getRadiation());
			}
		}
		if (hit.material->isReflective) {
			Vector reflectionDirection = hit.material->reflect(ray.rayDirection,
					hit.normalVector);
			reflectionDirection.Normalize();
			Ray reflectedRay = Ray(
					hit.intersectPoint + hit.normalVector * (0.001),
					reflectionDirection);
//			ray.rayDirection=-ray.rayDirection;

			outRadiance += trace(reflectedRay, depth + 1)
					* hit.material->Fresnel(ray.rayDirection, hit.normalVector);
		}
		if (hit.material->isRefractive) {

			Vector reflactionDirection = hit.material->refract(ray.rayDirection,
					hit.normalVector);
				reflactionDirection.Normalize();
			Ray reflactedRay = Ray(
					hit.intersectPoint + hit.normalVector * (0.0001),
					reflactionDirection);
			outRadiance += trace(reflactedRay, depth + 1)
					* (WHITE
							- hit.material->Fresnel(ray.rayDirection,
									hit.normalVector));

		}

		return outRadiance;

	}




	void writePixel(int x, int y, Color c) {
		image[y * screenWidth + x] = c;
	}

	void render() {
		for (int y = 0; y < screenHeight; y++) {
			for (int x = 0; x < screenWidth; x++) {
				Ray ray = camera.GetRay(x, y);
				Color pixelColor = trace(ray, 0);
				writePixel(x, y, pixelColor);
			}
		}
	}

	void SetAmbientColor(Color c) {
		this->ambientColor = c;
	}

	void SetCamera(MyCamera & newCam) {
		this->camera = newCam;
	}

	void AddObject(Intersectable *newObject) {

		objects.Add(newObject);
	}

	void AddLight(LightSource *newLight) {
		lightSources.Add(newLight);
	}

};

myMatrix getTheNewCoordSys(Hit intersect_point) {
	Vector new_y_axis = intersect_point.normalVector;
	new_y_axis.Normalize();
	Vector new_x_axis = Vector(0, 1, 0, 0) % intersect_point.normalVector;
	new_x_axis.Normalize();
	Vector new_z_axis = new_x_axis % new_y_axis;
	new_z_axis.Normalize();
	Vector new_origo = intersect_point.intersectPoint;

	myMatrix new_coord_sys = myMatrix(new_x_axis.x, new_x_axis.y, new_x_axis.z,
			0, new_y_axis.x, new_y_axis.y, new_y_axis.z, 0, new_z_axis.x,
			new_z_axis.y, new_z_axis.z, 0, new_origo.x, new_origo.y,
			new_origo.z, 1);

	return new_coord_sys;

}
Scene scene = Scene();

// Inicializacio, a program futasanak kezdeten, az OpenGL kontextus letrehozasa utan hivodik meg (ld. main() fv.)
void onInitialization() {
	glViewport(0, 0, screenWidth, screenHeight);

	Color ks = Color(0.4, 0.4, 0.4);
	Color kd = Color(255, 215, 0) / 255;
	Color k = Color(3.1, 2.7, 1.9);
	Color k_1 = Color(1.1, 1.7, 0.9);
	Color n = Color(0.17, 0.35, 1.5);
	Color k_glass = Color(0, 0, 0);
	Color n_glass = Color(1.5, 1.5, 1.5);
	Material *material = new Material(ks, Color(255, 200, 50) / 255, n, k,
			Color(), 30, true, false);
	Material *material_2 = new Material(ks, Color(205, 127, 50) / 255, n, k,
			Color(), 50, false, false);
	Material *material_3 = new Material(ks, Color(0, 178, 89) / 255, n, k,
			Color(), 50, false, false);
	Material *material_thirdplane = new Material(ks, Color(200, 200, 0) / 255, n, k,
				Color(), 50, false, false);
	Material *material_fourthplane = new Material(ks, Color(200, 0, 0) / 255, n, k,
				Color(), 50, false, false);
	Material *material_fifthplane = new Material(ks, Color(0, 0, 200) / 255, n, k,
					Color(), 50, false, false);
	Material *material_sixth = new Material(ks, Color(0, 220, 0) / 255, n, k,
						Color(), 50, false, false);

	Material *material_4 = new Material(Color(0,0,0), Color(0, 0, 0) / 255,
			Color(1.5,1.5,1.5), Color(), Color(), 30, true,  true);
//	Sphere implementation----------------------------------------------
	Sphere *firstSphere = new Sphere(material_4);

//	Ellipsoid implementation-------------------------------------------
	Ellipsoid *firstEllipsoid = new Ellipsoid(material_4);
	myMatrix transfom1 = myMatrix(0.15, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.15, 0, 0,
			0, 0, 1);
	myMatrix transfom3 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.2, 0,
			0.5, 1);
	myMatrix transfom2 = myMatrix(cos(M_PI / 6), sin(M_PI / 6), 0, 0,
			-sin(M_PI / 6), cos(M_PI / 6), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	myMatrix transform4 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0,
			1);

	Vector normal = Vector(-0.529863, 0.253724, 0.80924, 0);
	Vector origo = Vector(-0.150579, 0.20029, 0.229974, 1);

	Hit hit = Hit();
	hit.normalVector = normal;
	hit.intersectPoint = origo;

	myMatrix tr1 = myMatrix(0.15, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.15, 0, 0, 0,
			0, 1);
	myMatrix tr3 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.25, 0, 1);
	myMatrix tr = getTheNewCoordSys(hit);

	Ellipsoid *secondEllipsoid = new Ellipsoid(material);
	myMatrix first_connection_matrix = tr1 * tr3 * tr;
	firstEllipsoid->setTrasformationMatrix(transfom1);
	secondEllipsoid->setTrasformationMatrix(first_connection_matrix);

	myMatrix first_connection_matrix_inv = first_connection_matrix.inverse();
	myMatrix first_connection_matrix_inv_trnsp =
			first_connection_matrix_inv.Transp();
//	Vector normal_2 = normal * first_connection_matrix_inv_trnsp;
	Vector normal_2 = normal * first_connection_matrix;
	normal_2.Normalize();
	Vector origo_2 = origo * first_connection_matrix;

	hit.normalVector = normal_2;
	hit.intersectPoint = origo_2;

	tr1 = myMatrix(0.15 / 2, 0, 0, 0, 0, 0.25 / 2, 0, 0, 0, 0, 0.15 / 2, 0, 0,
			0, 0, 1);
//	tr1 = tr1.inverse();
	tr3 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.23, 0, 1);

	tr = getTheNewCoordSys(hit);
	Ellipsoid *thirdEllipsoid = new Ellipsoid(material);
	myMatrix second_connection_matrix = tr1 * tr3 * tr;
	thirdEllipsoid->setTrasformationMatrix(second_connection_matrix);

//	Cylinder implementation---------------------------------------------
	Cylinder *firstCylinder = new Cylinder(material);
	transfom1 = myMatrix(0.2, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 1);
	transfom3 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.4, 0, -0.6, 1);
	transfom2 = myMatrix(cos(M_PI / 2), sin(M_PI / 2), 0, 0, -sin(M_PI / 2),
			cos(M_PI / 2), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	firstCylinder->setTrasformationMatrix((transfom1 * transfom2) * transfom3);

//	Paraboloid implemenation--------------------------------------------
	Paraboloid *firstParaboloid = new Paraboloid(material);
	transfom1 = myMatrix(0.2, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 1);
	transfom3 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.4, 0, -0.6, 1);
	transfom2 = myMatrix(cos(M_PI / 2), sin(M_PI / 2), 0, 0, -sin(M_PI / 2),
			cos(M_PI / 2), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	firstParaboloid->setTrasformationMatrix((transfom1) * transfom3);

//	Plane implementation------------------------------------------------
	Plane_XZ *firstPlane = new Plane_XZ(material_2);
	Plane_XZ *fourthPlane = new Plane_XZ(material_fourthplane);
	Plane_XY *secondPlane = new Plane_XY(material_3);
	Plane_XY *thirdPlane = new Plane_XY(material_thirdplane);
	Plane_YZ *fifthPlane = new Plane_YZ(material_fifthplane);
	Plane_YZ *sixthPlane = new Plane_YZ(material_sixth);
	Plane_XZ_Down *planeDown = new Plane_XZ_Down(material_2);
	Plane_XZ_Up *planeUp = new Plane_XZ_Up(material_fourthplane);
	Plane_YZ_Right *planeRight = new Plane_YZ_Right(material_fifthplane);
	Plane_YZ_Left *planeLeft = new Plane_YZ_Left(material_sixth);
	Plane_XY_Front *planeFront = new Plane_XY_Front(material_3);
	Plane_XY_Back *planeBack = new Plane_XY_Back(material_thirdplane);




	myMatrix transfom_plane2 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
			0, -0.5, 1);
	myMatrix transfom_plane3 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
				0, -1.5, 1);
	myMatrix transfom_plane = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
				-1.5, 0, 1);
	myMatrix transfom_plane4 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
					-0.5, 0, 1);
	myMatrix transfom_plane5 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.5,
						0, 0, 1);
	myMatrix transfom_plane6 = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.5,
							0, 0, 1);
	firstPlane->setTrasformationMatrix(transfom_plane);
	secondPlane->setTrasformationMatrix(transfom_plane2);
	thirdPlane->setTrasformationMatrix(transfom_plane3);
	fourthPlane->setTrasformationMatrix(transfom_plane4);
	fifthPlane->setTrasformationMatrix(transfom_plane5);
	sixthPlane->setTrasformationMatrix(transfom_plane6);
//	scene.AddObject((Intersectable*) firstPlane);
//	scene.AddObject((Intersectable*) firstCylinder);
//	scene.AddObject((Intersectable*) firstSphere);
	scene.AddObject((Intersectable*) firstEllipsoid);
//	scene.AddObject((Intersectable*) secondEllipsoid);
//	scene.AddObject((Intersectable*) thirdEllipsoid);
//	scene.AddObject((Intersectable*) secondPlane);
//	scene.AddObject((Intersectable*) thirdPlane);
//	scene.AddObject((Intersectable*) fourthPlane);
//	scene.AddObject((Intersectable*) fifthPlane);
//	scene.AddObject((Intersectable*) sixthPlane);
		scene.AddObject((Intersectable*) planeDown);
		scene.AddObject((Intersectable*) planeUp);
		scene.AddObject((Intersectable*) planeRight);
		scene.AddObject((Intersectable*) planeLeft);
		scene.AddObject((Intersectable*) planeFront);
		scene.AddObject((Intersectable*) planeBack);





//	scene.AddObject((Intersectable*) firstParaboloid);

	scene.SetAmbientColor(Color(150, 150, 150) / 255);

//	PositionLightSource *light = new PositionLightSource(Vector(-1, 6, 7),
//			Color(1, 1, 1));
	PositionLightSource *light = new PositionLightSource(Vector(0.45, 0.45, 0.45),
			Color(0.5, 0.5, 0.5));

//	PositionLightSource *light = new PositionLightSource(Vector(-1, 0.8, 2),
//				Color(1, 1, 1));

	scene.AddLight((LightSource*) light);

//	MyCamera camera = MyCamera(Vector(2, 2, 5), Vector(0, 0, 0),
//			Vector(0, 1, 0));
	MyCamera camera = MyCamera(Vector(-0.25, 0, 0.25), Vector(0, 0, -0.25),
			Vector(0, 1, 0));

	scene.SetCamera(camera);
	scene.render();



}

// Rajzolas, ha az alkalmazas ablak ervenytelenne valik, akkor ez a fuggveny hivodik meg
void onDisplay() {
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);		// torlesi szin beallitasa
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // kepernyo torles


// Peldakent atmasoljuk a kepet a rasztertarba
	glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);
// Majd rajzolunk egy kek haromszoget
//	glColor3f(0, 0, 1);
//	glBegin(GL_TRIANGLES);
//	glVertex2f(-0.2f, -0.2f);
//	glVertex2f(0.2f, -0.2f);
//	glVertex2f(0.0f, 0.2f);
//	glEnd();

// ...

	glutSwapBuffers();     				// Buffercsere: rajzolas vege

	Vector v = Vector(1, 1, 1, 1);
	myMatrix m = myMatrix(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1);
	myMatrix m2 = myMatrix(1, 0, 0, 2, 0, 4, 0, 4, 0, 0, 4, 9, 0, 0, 0, 1);

	myMatrix transfom = myMatrix(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			1);
//	transfom.printM();
	myMatrix tr_in = transfom.inverse();

//	Vector v_2 = matrix_vector_multi(m, v);
	Vector v_3 = matrix_vector_multi(tr_in, v);
	Vector v1 = Vector(-2,-2,-5);
	Vector v2 = Vector(0,0,1);
	Vector v3 = v1%v2;
	v3.printOut();






}

// Billentyuzet esemenyeket lekezelo fuggveny (lenyomas)
void onKeyboard(unsigned char key, int x, int y) {
	if (key == 'd')
		glutPostRedisplay(); 		// d beture rajzold ujra a kepet

}

// Billentyuzet esemenyeket lekezelo fuggveny (felengedes)
void onKeyboardUp(unsigned char key, int x, int y) {

}

// Eger esemenyeket lekezelo fuggveny
void onMouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) // A GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON illetve GLUT_DOWN / GLUT_UP
		glutPostRedisplay(); 				// Ilyenkor rajzold ujra a kepet
}

// Eger mozgast lekezelo fuggveny
void onMouseMotion(int x, int y) {

}

// `Idle' esemenykezelo, jelzi, hogy az ido telik, az Idle esemenyek frekvenciajara csak a 0 a garantalt minimalis ertek
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);	// program inditasa ota eltelt ido

}

// ...Idaig modosithatod
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A C++ program belepesi pontja, a main fuggvenyt mar nem szabad bantani
int main(int argc, char **argv) {
	glutInit(&argc, argv); 				// GLUT inicializalasa
	glutInitWindowSize(600, 600); // Alkalmazas ablak kezdeti merete 600x600 pixel
	glutInitWindowPosition(100, 100); // Az elozo alkalmazas ablakhoz kepest hol tunik fel
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH); // 8 bites R,G,B,A + dupla buffer + melyseg buffer

	glutCreateWindow("Grafika hazi feladat"); // Alkalmazas ablak megszuletik es megjelenik a kepernyon

	glMatrixMode(GL_MODELVIEW);	// A MODELVIEW transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);// A PROJECTION transzformaciot egysegmatrixra inicializaljuk
	glLoadIdentity();

	onInitialization();			// Az altalad irt inicializalast lefuttatjuk

	glutDisplayFunc(onDisplay);				// Esemenykezelok regisztralasa
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();					// Esemenykezelo hurok

	return 0;
}

