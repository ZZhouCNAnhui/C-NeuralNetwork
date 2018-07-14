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
        public static Matrix operator +(Matrix lhs, Matrix rhs)
        {
            if (lhs.yLength != rhs.yLength || rhs.xLength != 1)
                throw new Exception("无法将矩阵相加");
            Matrix re = lhs;
            for (int i = 0; i < re.xLength; i++)
                for (int j = 0; j < re.yLength; j++)
                    re.matrix[i, j] += rhs.matrix[0, j];
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

    }

    public static class Function
    {

        public static void Multiply(ref Matrix v, float n)
        {
            for (int i = 0; i < v.xLength; i++)
                for (int j = 0; j < v.yLength; j++)
                    v.vMatrix[i, j] *= n;
        }
        public static void Add(ref Matrix lhs, Matrix rhs)
        {
            if (lhs.xLength != rhs.xLength || lhs.yLength != rhs.yLength)
                throw new Exception("无法将矩阵相加");
            for (int i = 0; i < lhs.xLength; i++)
                for (int j = 0; j < lhs.yLength; j++)
                    lhs.vMatrix[i, j] += rhs.vMatrix[i, j];
        }
        public static void Add(ref Matrix lhs, float rhs)
        {
            for (int i = 0; i < lhs.xLength; i++)
                for (int j = 0; j < lhs.yLength; j++)
                    lhs.vMatrix[i, j] += rhs;
        }
        public static void Sub(ref Matrix lhs, Matrix rhs)
        {
            if (lhs.xLength != rhs.xLength || lhs.yLength != rhs.yLength)
                throw new Exception("无法将矩阵相减");
            for (int i = 0; i < lhs.xLength; i++)
                for (int j = 0; j < lhs.yLength; j++)
                    lhs.vMatrix[i, j] -= rhs.vMatrix[i, j];
        }
        public static void Sub(ref Matrix lhs, float rhs)
        {
            for (int i = 0; i < lhs.xLength; i++)
                for (int j = 0; j < lhs.yLength; j++)
                    lhs.vMatrix[i, j] -= rhs;
        }
        public static void Power(ref Matrix v, float n)
        {
            v.Foreach(a => { return (float)Math.Pow((double)a, (double)n); });
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

namespace NeuralNetwork
{
    using matrix;
    using ActivationFunctions;

    namespace Layers
    {
        using NeuralNetwork;

        /// <summary>
        /// 默认的神经层，使用矩阵
        /// </summary>
        public class Denes : Layer<Matrix, Matrix>
        {
            Random r = new Random();
            /// <summary>
            /// 构造神经层
            /// </summary>
            /// <param name="inputsize">输入数据大小</param>
            /// <param name="outputsize">输出数据大小</param>
            /// <param name="Name">神经层名字</param>
            /// <param name="af">激励函数</param>
            /// <param name="WeightRa">是否权重随机</param>
            public Denes(int inputsize, int outputsize, string Name,AF af =null,bool WeightRa=true) : 
                base(Name,af)
            {

                if (WeightRa)
                    Weight = matrix.Function.Random(inputsize, outputsize, -1, 1);
                else
                    Weight = new Matrix(inputsize, outputsize, 2);
                Bias = new Matrix(1, outputsize, 0);
            }
            public override Matrix Function(Matrix input)
            {
                Matrix M = Weight * input + Bias;
                if (ActivationFunction != null)
                    M.Activation(ActivationFunction);
                return M;
            }

            public override Matrix FunctionNoAF(Matrix input)
            {
                Matrix M = Weight * input + Bias;
                return M;
            }
        }

        /// <summary>
        /// 默认的神经层的缩小，使用单个数
        /// </summary>
        public class SmallDenes : Layer<Matrix,float>
        {
            Random r = new Random();
            public SmallDenes(string Name,AF af=null):base(Name,af)
            {
                Weight = (float)r.NextDouble();
                Bias = 0;

            }
            public override Matrix Function(Matrix input)
            {
                Matrix ou = input;
                matrix.Function.Multiply(ref ou, Weight);
                matrix.Function.Add(ref ou, Bias);
                return ou;
            }

            public override Matrix FunctionNoAF(Matrix input)
            {
                Matrix ou = input;
                matrix.Function.Multiply(ref ou, Weight);
                matrix.Function.Add(ref ou, Bias);
                return ou;
            }
        }

    }

    /// <summary>
    /// 激励函数
    /// </summary>
    namespace ActivationFunctions
    {
        /// <summary>
        /// 激励函数类型
        /// </summary>
        /// <param name="input">输出数据</param>
        /// <param name="Derivatives">是否为导函数</param>
        /// <returns>输出结果</returns>
        public delegate float AF(float input, bool Derivatives);
        public static class ActivationFunctions
        {
            public static AF Relu = delegate (float input, bool Derivatives)
            {
                if (Derivatives)
                {
                    if (input > 0)
                        return 1;
                    else
                        return 0;
                }
                else
                {
                    return Math.Max(0, input);
                }
            };

            public static AF Sigmoid = delegate (float input, bool Derivatives)
            {
                if (Derivatives)
                {
                    return (float)((1 / (1 + Math.Pow(Math.E, -input))) * (1 - (1 / (1 + Math.Pow(Math.E, -input)))));
                }
                else
                {
                    return (float)(1 / (1 + Math.Pow(Math.E, -input)));
                }
            };
        }
    }


    public abstract class Layer<IO, WB>
    {
        public Layer(string Name, AF af)
        {
            this.Name = Name;
            this.ActivationFunction = af;
        }
        public int LayerNumber;
        public AF ActivationFunction;
        public WB Weight;
        public WB Bias;
        public abstract IO Function(IO input);
        public abstract IO FunctionNoAF(IO input);
        public string Name;
        public string LinkLayerName;

    }


    public class Constant<T1, T2> : Layer<T1, T2>
    {
        public Constant(string Name) : base(Name,null) { }

        public override T1 Function(T1 input)
        {
            return input;
        }

        public override T1 FunctionNoAF(T1 input)
        {
            return input;
        }
    }

    /// <summary>
    /// 神经网络
    /// </summary>
    /// <typeparam name="Tensor">输出输出数据的类型,现只支持Matrix</typeparam>
    /// <typeparam name="LayerWB">神经网络中权重和偏置的类型,现只支持Matrix</typeparam>
    public class Network<Tensor, LayerWB>where Tensor: Matrix
    {
        private Dictionary<string, Layer<Tensor, LayerWB>> Structure;

        private Constant<Tensor, LayerWB> Input;
        private Constant<Tensor, LayerWB> Output;

        public Network()
        {
            Input = new Constant<Tensor, LayerWB>("Input");
            Output = new Constant<Tensor, LayerWB>("Output");
            Structure = new Dictionary<string, Layer<Tensor, LayerWB>>
            {
                { "Input", Input },
                { "Output", Output }
            };

        }

        /// <summary>
        /// 在神经网络中添加一个层
        /// </summary>
        /// <param name="layer">继承自Layer的一个层</param>
        /// <param name="LinkLayer">此层输出传递给的层的名字</param>
        public void AddLayer(Layer<Tensor, LayerWB> layer, string LinkLayer)
        {
            Structure.Add(layer.Name, layer);
            Structure[layer.Name].LinkLayerName = LinkLayer;
            string link = "Input";
            for (int i = 0; i < Structure.Count-1; i++)
            {
                Layer<Tensor, LayerWB> la = Structure[link];
                la.LayerNumber = i;
                link = la.LinkLayerName;
            }
            Output.LayerNumber = Structure.Count;
        }

        /// <summary>
        /// 设置Tensor输入传递给那一层
        /// </summary>
        /// <param name="LinkLayer">Tensor输入传递给的层的名字</param>
        public void SetInputLink(string LinkLayer)
        {
            Input.LinkLayerName = LinkLayer;
        }

        /// <summary>
        /// 运行神经网络
        /// </summary>
        /// <param name="T">输入数据</param>
        /// <returns>输出Output层的输出</returns>
        public Tensor RunNetwork(Tensor T)
        {
            Tensor data = T;
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> layer = Structure[link];
                data = layer.Function(data);
                link = layer.LinkLayerName;
            }
            return data;
        }

        public Tensor RunForVariable(Tensor T,int LayerNumber)
        {
            Tensor data = T;
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> layer = Structure[link];
                data = layer.Function(data);
                if (layer.LayerNumber == LayerNumber)
                    return data;
                link = layer.LinkLayerName;
            }
            return data;
        }
        public Layer<Tensor, LayerWB> FindLayerBynum(int LayerNumber)
        {
            string link = "Input";
            for (int i = 0; i < Structure.Count - 1; i++)
            {
                Layer<Tensor, LayerWB> la = Structure[link];
                if (la.LayerNumber == LayerNumber)
                    return la;
                link = la.LinkLayerName;
            }
            return null;
        }

        /// <summary>
        /// 训练神经网络
        /// </summary>
        /// <param name="XData">输入数据</param>
        /// <param name="YData">输出数据</param>
        /// <param name="trainSetup">训练次数</param>
        /// <param name="isShow">是否显示过程</param>
        public void Train(Tensor XData,Tensor YData,int trainSetup,bool isShow=true)
        {
            List<Tensor> tensors = new List<Tensor>() { XData };
            string link = "Input";
            for (int i = 0; i < Structure.Count; i++)
            {
                Layer<Tensor, LayerWB> layer = Structure[link];
                tensors.Add(layer.Function(tensors[i]));
                link = layer.LinkLayerName;
            }
            Matrix Out = tensors[tensors.Count - 1];
            Function.Sub(ref Out,YData);
            Function.Power(ref Out, 2);
            float loss = Function.Mean(Out);
            Console.WriteLine("loss:"+loss);

            Matrix Dloss = tensors[tensors.Count - 1];
            Function.Sub(ref Dloss, YData);
            Function.Multiply(ref Dloss, 2);


            Tensor Dlayer = RunForVariable(XData, Structure.Count - 2);
            Tensor Dzl = Dlayer;
            Dlayer = FindLayerBynum(Structure.Count - 1).FunctionNoAF(Dlayer);
            Dlayer.Derivatives(FindLayerBynum(Structure.Count - 1).ActivationFunction);




            //for (int i = tensors.Count; i >= 0; i--)
            //{
            //    tensors[i]
            //}

        }
    }

}