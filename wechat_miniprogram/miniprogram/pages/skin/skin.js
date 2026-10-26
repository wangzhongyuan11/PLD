var api = require("../../utils/api.js");

Page({
  data: {
    picked: null,   // {path, name, size}
    thickness: 2,
    submitting: false,
  },

  chooseImage: function () {
    var self = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["album", "camera"],
      success: function (res) {
        var f = res.tempFiles && res.tempFiles[0];
        if (!f) return;
        if (f.size > 8 * 1024 * 1024) {
          wx.showToast({ title: "图片需在 8MB 以内", icon: "none" });
          return;
        }
        self.setData({
          picked: { path: f.tempFilePath, name: "photo.jpg", size: f.size },
        });
      },
      fail: function (err) { api.pickFailToast("选择照片", err); },
    });
  },

  onThickness: function (e) {
    this.setData({ thickness: Number(e.detail.value) });
  },

  clearImage: function () {
    this.setData({ picked: null });
  },

  submit: function () {
    var self = this;
    if (!this.data.picked || this.data.submitting) return;
    this.setData({ submitting: true });
    api.submitTaskFlow({
      type: "skin",
      files: [{ path: this.data.picked.path, name: this.data.picked.name }],
      params: { thickness: this.data.thickness },
      title: this.data.picked.name,
    }).then(function (id) {
      wx.redirectTo({ url: "/pages/task/task?id=" + id });
    }).catch(function (err) {
      self.setData({ submitting: false });
      wx.showToast({ title: err.message || "提交失败", icon: "none" });
    });
  },
});
