using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;
using System.Collections;

namespace matrix
{
    using NeuralNetwork.ActivationFunctions;

    public class Matrix : IEnumerable<float>
    {
        public Matrix(params int[] v)
        {


            if (v.Length != 3)
                throw new Exception("参数数量错误");
            xLength = v[0];
            yLength = v[1];
            matrix = new float[xLength, yLength];
            for (int i = 0; i < xLength; i++)
                for (int j = 0; j < yLength; j++)
                    matrix[i, j] = v[2];
        }


        public int xLength;
        public int yLength;
        private float[,] matrix;
        public string Shape { get { return string.Format("({0} {1})", xLength, yLength); } }
        public float[,] vMatrix { get { return matrix; } set { matrix = value; } }

        public void Foreach(Func<float, float> func)
        {
            for (int j = 0; j < yLength; j++)
                for (int i = 0; i < xLength; i++)
                    matrix[i, j] = func(matrix[i, j]);
        }
        public void Activation(AF af)
        {
            for (int j = 0; j < yLength; j++)
                for (int i = 0; i < xLength; i++)
                    matrix[i, j] = af(matrix[i, j], false);
        }
        public void Derivatives(AF af)
        {
            for (int j = 0; j < yLength; j++)
                for (int i = 0; i < xLength; i++)
                    matrix[i, j] = af(matrix[i, j], true);
        }
        public void ReShape(int x,int y)
        {
            if (x * y != xLength * yLength)
                throw new Exception("无效形状");
            float[,] Newmatrix = new float[x, y];
            List<float> li = new List<float>();
            foreach (var item in this)
                li.Add(item);
            xLength = x;
            yLength = y;
            int xp = 0, yp = 0;
            foreach (var item in li)
            {
                Newmatrix[xp, yp] = item;
                xp++;
                if(xp >= xLength)
                {
                    xp = 0;
                    yp++;
                }
            }
            matrix = Newmatrix;

        }
        public Matrix Clone()
        {
            Matrix re = new Matrix(xLength, yLength, 0);
            
            for (int i = 0; i < xLength; i++)
                for (int j = 0; j < yLength; j++)
                    re.vMatrix[i, j] = vMatrix[i, j];
            return re;
        }

        public override string ToString()
        {
            string s = "[";

            for (int j = 0; j < yLength; j++)
            {
                s += "[";
                for (int i = 0; i < xLength; i++)
                    s += matrix[i, j].ToString() + " ";
                if (j != yLength - 1)
                    s += "]\n ";
            }
            s += "]]";
            return s;
        }
        public IEnumerator<float> GetEnumerator()
        {
            for (int j = 0; j < yLength; j++)
                for (int i = 0; i < xLength; i++)
                    yield return matrix[i, j];
        }
        IEnumerator IEnumerable.GetEnumerator()
        {
            return null;
        }
        public Matrix this[int x, int y]
        {
            get
            {
                if (x == -1 && y == -1) return this;
                Matrix m = null;
                if (x == -1)
                {
                    m = new Matrix(xLength, 1, 0);
                    for (int i = 0; i < xLength; i++)
                        m.matrix[i, 0] = this.matrix[i, y];
                }
                else if (y == -1)
                {
                    m = new Matrix(1, yLength, 0);
                    for (int i = 0; i < yLength; i++)
                        m.matrix[0, i] = this.matrix[x, i];
                }
                else
                {
                    m = new Matrix(1, 1);
                    m.matrix[0, 0] = this.matrix[x, y];
                }
                return m;
            }
            set
            {
                if (x == -1 && y == -1) this.matrix = value.matrix;
                if (x == -1)
                {
                    for (int i = 0; i < xLength; i++)
                        this.matrix[i, y] = value.matrix[i, 0];
                }
                else if (y == -1)
                {
                    for (int i = 0; i < yLength; i++)
                        this.matrix[x, i] = value.matrix[0, i];
                }
                else
                {
                    this.matrix[x, y] = value.matrix[0, 0];
                }
            }
        }
        public static Matrix operator -(Matrix lhs, float rhs)
        {
            Matrix re = new Matrix(lhs.xLength, lhs.yLength, 0);
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.matrix[i, j] = lhs.vMatrix[i, j] - rhs;
            return re;
        }
        public static Matrix operator -(Matrix lhs, Matrix rhs)
        {
            if (lhs.yLength != rhs.yLength || rhs.xLength != lhs.xLength)
                throw new Exception("无法将矩阵相减");
            Matrix re = new Matrix(rhs.xLength, rhs.yLength, 0);
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.matrix[i, j] = lhs.vMatrix[i, j] - rhs.matrix[i, j];
            return re;
        }
        public static Matrix operator +(Matrix lhs, Matrix rhs)
        {
            if (lhs.yLength != rhs.yLength || rhs.xLength != 1)
                throw new Exception("无法将矩阵相加");
            Matrix re = new Matrix(rhs.xLength, rhs.yLength,0);
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.matrix[i, j] = rhs.matrix[0, j]+lhs.vMatrix[i,j];
            return re;
        }
        public static Matrix operator +(Matrix lhs, float rhs)
        {
            Matrix re = new Matrix(lhs.xLength, lhs.yLength, 0);
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.matrix[i, j] = lhs.vMatrix[i, j] + rhs;
            return re;
        }
        public static Matrix operator *(Matrix lhs, Matrix rhs)
        {
            if (lhs.xLength != rhs.yLength)
                throw new Exception("无法将矩阵相乘");
            Matrix re = new Matrix(rhs.xLength, lhs.yLength, 0);

            for (int k = 0; k < lhs.xLength; k++)
            {
                Matrix men = new Matrix(rhs.xLength, lhs.yLength, 0);
                for (int i = 0; i < rhs.xLength; i++)
                {
                    Matrix vs = lhs[k, -1];
                    Function.Multiply(ref vs, rhs.matrix[i, k]);
                    Matrix a = men[i, -1];
                    Function.Add(ref a, vs);
                    men[i, -1] = a;
                }
                Function.Add(ref re, men);
            }
            return re;
        }
        public static Matrix operator *(Matrix lhs, float rhs)
        {
            Matrix re = new Matrix(lhs.xLength, lhs.yLength, 0);
            for (int i = 0; i < lhs.xLength; i++)
                for (int j = 0; j < lhs.yLength; j++)
                    re.vMatrix[i, j] = rhs * lhs.vMatrix[i,j];
            return re;
        }

    }

    public static class Function
    {

        public static void Multiply(ref Matrix v, float n)
        {
            for (int i = 0; i < v.xLength; i++)
                for (int j = 0; j < v.yLength; j++)
                    v.vMatrix[i, j] *= n;
        }
        public static void Multiply(ref Matrix v, Matrix n)
        {
            if (v.xLength != n.xLength || v.yLength != n.yLength)
                throw new Exception("无法将矩阵相乘");
            for (int i = 0; i < v.xLength; i++)
                for (int j = 0; j < v.yLength; j++)
                    v.vMatrix[i, j] *= n.vMatrix[i,j];
        }
        public static Matrix Multiply(Matrix v, float n)
        {
            Matrix re = new Matrix(v.xLength, v.yLength, 0);
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.vMatrix[i, j] = n * v.vMatrix[i,j];
            return re;
        }
        public static Matrix Multiply(Matrix v, Matrix n)
        {
            if (v.xLength != n.xLength || v.yLength != n.yLength)
                throw new Exception("无法将矩阵相乘");
            Matrix re = new Matrix(v.xLength, v.yLength, 0);
            for (int i = 0; i < v.xLength; i++)
                for (int j = 0; j < v.yLength; j++)
                    re.vMatrix[i,j] = v.vMatrix[i, j] * n.vMatrix[i, j];
            return re;
        }

        public static void Add(ref Matrix m, Matrix v)
        {
            if (m.xLength != v.xLength || m.yLength != v.yLength)
                throw new Exception("无法将矩阵相加");
            for (int i = 0; i < m.xLength; i++)
                for (int j = 0; j < m.yLength; j++)
                    m.vMatrix[i, j] += v.vMatrix[i, j];
        }
        public static void Add(ref Matrix v, float n)
        {
            for (int i = 0; i < v.xLength; i++)
                for (int j = 0; j < v.yLength; j++)
                    v.vMatrix[i, j] += n;
        }
        public static Matrix Add(Matrix v1, Matrix v2)
        {
            if (v1.xLength != v2.xLength || v1.yLength != v2.yLength)
                throw new Exception("无法将矩阵相加");
            Matrix re = new Matrix(v1.xLength, v1.yLength, 0);
            for (int i = 0; i < v1.xLength; i++)
                for (int j = 0; j < v1.yLength; j++)
                    re.vMatrix[i,j] =v1.vMatrix[i, j] + v2.vMatrix[i, j];
            return re;
        }
        public static Matrix Add( Matrix v, float n)
        {
            Matrix re = new Matrix(v.xLength, v.yLength, 0);
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.vMatrix[i, j] = n + v.vMatrix[i,j];
            return re;
        }
        public static float AddAll(Matrix lhs)
        {
            float sum = 0;
            foreach (var item in lhs)
                sum += item;
            return sum;
        }

        public static void Sub(ref Matrix m, Matrix v)
        {
            if (m.xLength != v.xLength || m.yLength != v.yLength)
                throw new Exception("无法将矩阵相减");
            for (int i = 0; i < m.xLength; i++)
                for (int j = 0; j < m.yLength; j++)
                    m.vMatrix[i, j] -= v.vMatrix[i, j];
        }
        public static void Sub(ref Matrix v, float n)
        {
            for (int i = 0; i < v.xLength; i++)
                for (int j = 0; j < v.yLength; j++)
                    v.vMatrix[i, j] -= n;
        }
        public static Matrix Sub(Matrix v1, Matrix v2)
        {
            if (v1.xLength != v2.xLength || v1.yLength != v2.yLength)
                throw new Exception("无法将矩阵相加");
            Matrix re = new Matrix(v1.xLength, v1.yLength, 0);
            for (int i = 0; i < v1.xLength; i++)
                for (int j = 0; j < v1.yLength; j++)
                    re.vMatrix[i, j] = v1.vMatrix[i, j] - v2.vMatrix[i, j];
            return re;
        }
        public static Matrix Sub(Matrix v, float n)
        {
            Matrix re = new Matrix(v.xLength, v.yLength, 0);
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.vMatrix[i, j] = v.vMatrix[i, j] - n;
            return re;
        }

        public static void Power(ref Matrix v, float n)
        {
            v.Foreach(a => { return (float)Math.Pow((double)a, (double)n); });
        }
        public static Matrix Power(Matrix v, float n)
        {
            Matrix re = new Matrix(v.xLength, v.yLength, 0);
            for (int i = 0; i < v.xLength; i++)
                for (int j = 0; j < v.yLength; j++)
                    re.vMatrix[i, j] = v.vMatrix[i, j];
            re.Foreach(a => { return (float)Math.Pow(a, n); });
            return re;
        }

        public static Matrix Range(int x, int y)
        {
            Matrix re = new Matrix(x, y, 0);
            int num = 0;
            for (int j = 0; j < re.yLength; j++)
            {
                for (int i = 0; i < re.xLength; i++)
                {
                    re.vMatrix[i, j] = num;
                    num++;
                }
            }
            return re;
        }
        public static Matrix LineSpace(float from, float end, int num = 25)
        {
            Matrix re = new Matrix(num, 1, 0);
            float n = from;
            float jiange = (end - from) / num;
            for (int i = 0; i < re.xLength; i++)
            {
                re.vMatrix[i, 0] = n;
                n += jiange;
            }

            return re;
        }
        public static Matrix Random(int x, int y, float min, float max)
        {

            Random r = new Random();
            Matrix re = new Matrix(x, y, 0);

            for (int j = 0; j < re.yLength; j++)
                for (int i = 0; i < re.xLength; i++)
                    re.vMatrix[i, j] = (max - min) * ((float)r.NextDouble()) + min;
            return re;
        }
        public static Matrix Random(int x, int y, float min, float max, int seed)
        {
            Random r = new Random(seed);
            Matrix re = new Matrix(x, y, 0);

            for (int j = 0; j < re.yLength; j++)
                for (int i = 0; i < re.xLength; i++)
                    re.vMatrix[i, j] = (max - min) * ((float)r.NextDouble()) + min;
            return re;
        }
        public static float Mean(Matrix vs)
        {
            float re = 0;
            foreach (var item in vs)
                re += item;
            re /= (vs.xLength * vs.yLength);
            return re;
        }
    }

}
