using Avalonia.Controls;
using OptiLearn.ViewModels;

namespace OptiLearn.Views
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            grCourses.DataContext = new Course("Zexsoft", "Impossible possibilities.");
        }
    }
}
