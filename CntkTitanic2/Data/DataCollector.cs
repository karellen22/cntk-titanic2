using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CntkTitanic2.Data
{
    public class DataCollector
    {
        public List<string[]> ReadData(string fileName)
        {
            var list = new List<string[]>();
            var count = 0;
            foreach (var line in File.ReadAllLines(fileName))
            {
                list.Add(line.Split(','));
                count++;
            }

            return list;
        }
    }
}
