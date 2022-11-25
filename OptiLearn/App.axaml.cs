using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Material.Colors;
using Material.Styles.Themes.Base;
using Material.Styles.Themes;
using OptiLearn.ViewModels;
using OptiLearn.Views;

namespace OptiLearn
{
    public partial class App : Application
    {
        public override void Initialize()
        {
            AvaloniaXamlLoader.Load(this);
        }

        public override void OnFrameworkInitializationCompleted()
        {
            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                desktop.MainWindow = new MainWindow
                {
                    DataContext = new MainWindowViewModel(),
                };
            }

            /*PrimaryColor primary = PrimaryColor.Indigo;
            Color primaryColor = SwatchHelper.Lookup[(MaterialColor)primary];

            SecondaryColor secondary = SecondaryColor.Teal;
            Color secondaryColor = SwatchHelper.Lookup[(MaterialColor)secondary];

            // For dark theme use  Theme.Dark;
            IBaseTheme baseTheme = Theme.Light;

            ITheme theme = Theme.Create(baseTheme, primaryColor, secondaryColor);*/

            base.OnFrameworkInitializationCompleted();
        }
    }
}
