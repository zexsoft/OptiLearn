using ReactiveUI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OptiLearn.ViewModels
{
    public class Course : ViewModelBase
    {
        public string Name;
        public string Content;

        public string txName
        {
            get => Name;
            set => this.RaiseAndSetIfChanged(ref Name, value);
        }

        public string txContent
        {
            get => Content;
            set => this.RaiseAndSetIfChanged(ref Content, value);
        }

        public Course() {
            Name = "Course";
            Content = string.Empty;
        }

        public Course(string name, string content)
        {
            Name = name;
            Content = content;
        }

        public static Course FromWiki(string url)
        {
            return new Course("Wiki", "Sample text.");
        }
    }
}
