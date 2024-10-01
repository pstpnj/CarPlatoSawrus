namespace GUI.Pages;
public partial class HomePage : FlyoutPage
{
	public HomePage()
	{
		InitializeComponent();
	}

	private async void OnClickedToFlyoutPage1(object sender, EventArgs e)
    {
		await Navigation.PushAsync(new ContentFlyoutPage1());
    }
}