try:
    from torchvision.models.optical_flow import raft_large

    print("✅ RAFT is available!")
    print("You can proceed to load the model.")

except Exception as e:
    print("❌ Error")
    print(e)