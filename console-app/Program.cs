// Program.cs

using System;
using Library;

namespace console_app
{
  class Program
  {
    static void Main(string[] args)
    {
        Console.WriteLine($"The answer is {new Thing().Get(19, 23)}");
        Console.WriteLine($"The answer is {new Thing().Get(19, 24)}");
    }
  }
}