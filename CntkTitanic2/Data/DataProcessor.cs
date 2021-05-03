using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CntkTitanic2.Data
{
    internal class DataProcessor
    {
        public DataProcessor()
        {
        }

        internal void ProcessData(List<string[]> data, List<float> inputs, List<float> output)
        {
            foreach (var item in data)
            {
                var floats = GetFloats(item);
                inputs.AddRange(floats.GetRange(1, 6));
                output.Add(floats[0]);
            }
        }

        private List<float> GetFloats(string[] item)
        {
            var survived = float.Parse(item[0]);
            var ticketClass = float.Parse(item[1]) / 10;
            var sex = item[3] == "female" ? 1.0f : 0.0f;
            var age = float.Parse(item[4], CultureInfo.InvariantCulture.NumberFormat) / 100;
            var siblingsSpouses = float.Parse(item[5]) / 100;
            var parentsChildren = float.Parse(item[6]) / 100;
            var farePrice = float.Parse(item[7], CultureInfo.InvariantCulture.NumberFormat) / 100;
            return new List<float> { survived, ticketClass, sex, age, siblingsSpouses, parentsChildren, farePrice };
        }
    }
}