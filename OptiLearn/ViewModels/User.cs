using Newtonsoft.Json.Linq;
using ReactiveUI;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OptiLearn.ViewModels
{
    public class User : ViewModelBase
    {
        public int Id = 0;
        public string Name = "User";
        public string PasswordHash = "0";

        public CultureInfo Region = CultureInfo.CurrentCulture;
        public int Points = 0;

        public ObservableCollection<OngoingCourse> Courses { get; set; } = new();
        public ObservableCollection<int> Friends { get; set; } = new();

        #region MVVM
        public int cntId
        {
            get => Id;
            set => this.RaiseAndSetIfChanged(ref Id, value);
        }

        public string txName
        {
            get => Name;
            set => this.RaiseAndSetIfChanged(ref Name, value);
        }

        public string txPassword
        {
            set => this.RaiseAndSetIfChanged(ref PasswordHash, value);
        }

        public string txRegion
        {
            get => Region.DisplayName;
            set => this.RaiseAndSetIfChanged(ref Region, new CultureInfo(value));
        }

        public int cntPoints
        {
            get => Points;
        }
        #endregion
    }
}
